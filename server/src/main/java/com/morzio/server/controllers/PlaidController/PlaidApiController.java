package com.morzio.server.controllers.PlaidController;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.morzio.server.dtos.PlaidService.InstallmentPlanDto;
import com.morzio.server.dtos.PlaidService.MLResponseDto;
import com.morzio.server.dtos.PlaidService.PlaidOnSuccessRequestDto;
import com.morzio.server.dtos.PlaidService.PlaidOnSuccessResponseDto;
import com.morzio.server.dtos.PlaidService.SelectPlanRequestDto;
import com.morzio.server.dtos.PlaidService.SelectPlanResponseDto;
import com.morzio.server.dtos.PlaidService.TransactionDto;
import com.morzio.server.entities.PaymentSession;
import com.morzio.server.entities.Installment;
import com.morzio.server.entities.InstallmentPlan;
import com.morzio.server.repositorys.PaymentSessionRepository;
import com.morzio.server.repositorys.Installments;
import com.morzio.server.repositorys.InstallmentsPlanRepository;
import com.morzio.server.services.MLService;
import com.morzio.server.services.PlaidService;
import com.plaid.client.model.AccountsGetResponse;
import com.plaid.client.model.Transaction;
import com.plaid.client.model.TransactionsGetResponse;

@RestController
@RequestMapping("/api/plaid")
public class PlaidApiController {

        private static final Logger logger = LoggerFactory.getLogger(PlaidApiController.class);

        @Autowired
        private PlaidService plaidService;

        @Autowired
        private MLService mlService;

        @Autowired
        private PaymentSessionRepository paymentSessionRepository;

        @Autowired
        private Installments installmentsRepository;

        @Autowired
        private InstallmentsPlanRepository installmentPlanRepository;

        /**
         * Handles the Plaid Link success callback
         * Exchanges public token, fetches account balances and transactions,
         * and returns installment plan options based on risk logic
         *
         * @param request Contains publicToken and transactionAmount
         * @return PlaidOnSuccessResponseDto with installment plans or error
         */
        @PostMapping("/on-success")
        public ResponseEntity<PlaidOnSuccessResponseDto> handlePlaidSuccess(
                        @RequestBody PlaidOnSuccessRequestDto request) {
                try {
                        // Validate request
                        if (request.getPublicToken() == null || request.getPublicToken().isEmpty()) {
                                return ResponseEntity.badRequest().body(
                                                new PlaidOnSuccessResponseDto(false, null, null, null,
                                                                "Public token is required"));
                        }

                        if (request.getTransactionAmount() == null || request.getTransactionAmount() <= 0) {
                                return ResponseEntity.badRequest().body(
                                                new PlaidOnSuccessResponseDto(false, null, null, null,
                                                                "Valid transaction amount is required"));
                        }

                        logger.info("Processing Plaid success callback for transaction amount: €{}",
                                        request.getTransactionAmount());

                        // Step 1: Exchange public token for access token
                        String accessToken = plaidService.exchangePublicToken(request.getPublicToken());
                        logger.info("Successfully exchanged public token for access token");

                        // Step 2: Fetch account balances
                        AccountsGetResponse accountsResponse = plaidService.getAccountBalances(accessToken);
                        logger.info("Retrieved {} account(s)", accountsResponse.getAccounts().size());

                        // Step 3: Fetch transactions
                        TransactionsGetResponse transactionsResponse = plaidService.getTransactions(accessToken);
                        List<TransactionDto> transactionDtos = mapTransactions(transactionsResponse.getTransactions());

                        logger.info("Retrieved {} transaction(s) from the last 90 days", transactionDtos.size());

                        // Log all transactions
                        logTransactions(transactionDtos);

                        // Step 4: Call ML service for real risk assessment
                        Double transactionAmount = request.getTransactionAmount();
                        logger.info("Calling ML service for risk assessment. Transaction amount: £{}",
                                        transactionAmount);

                        MLResponseDto mlResponse = mlService.predict(transactionDtos, transactionAmount);

                        if (mlResponse.isApproved() && mlResponse.getMax_installments() > 0) {
                                // ML approved - generate installment plans up to max allowed
                                List<InstallmentPlanDto> installmentPlans = generateInstallmentPlans(
                                                transactionAmount, mlResponse.getMax_installments());

                                logger.info("ML APPROVED - Max installments: {}, Offering {} plan options",
                                                mlResponse.getMax_installments(), installmentPlans.size());

                                return ResponseEntity.ok(
                                                new PlaidOnSuccessResponseDto(
                                                                true,
                                                                "Installment plans available (ML approved)",
                                                                installmentPlans,
                                                                transactionDtos,
                                                                null));
                        } else {
                                // ML declined
                                logger.warn("ML DECLINED - Risk assessment failed for transaction amount: £{}",
                                                transactionAmount);

                                return ResponseEntity.status(HttpStatus.PAYMENT_REQUIRED).body(
                                                new PlaidOnSuccessResponseDto(
                                                                false,
                                                                null,
                                                                null,
                                                                transactionDtos,
                                                                "Payment declined based on risk assessment"));
                        }

                } catch (IOException e) {
                        logger.error("IOException while processing Plaid token: {}", e.getMessage(), e);
                        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(
                                        new PlaidOnSuccessResponseDto(
                                                        false,
                                                        null,
                                                        null,
                                                        null,
                                                        "Failed to process Plaid token: " + e.getMessage()));
                } catch (Exception e) {
                        logger.error("Unexpected error while processing Plaid callback: {}", e.getMessage(), e);
                        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(
                                        new PlaidOnSuccessResponseDto(
                                                        false,
                                                        null,
                                                        null,
                                                        null,
                                                        "An unexpected error occurred: " + e.getMessage()));
                }
        }

        /**
         * Selects a plan and initiates payment
         *
         * @param request Contains sessionId and downPaymentAmount
         * @return SelectPlanResponseDto with paymentId
         */
        @PostMapping("/select-plan")
        public ResponseEntity<SelectPlanResponseDto> selectPlan(@RequestBody SelectPlanRequestDto request) {
                try {
                        if (request.getSessionId() == null || request.getSessionId().isEmpty()) {
                                return ResponseEntity.badRequest().body(
                                                new SelectPlanResponseDto(false, null, "Session ID is required"));
                        }

                        if (request.getDownPaymentAmount() == null || request.getDownPaymentAmount() <= 0) {
                                return ResponseEntity.badRequest().body(
                                                new SelectPlanResponseDto(false, null,
                                                                "Valid down payment amount is required"));
                        }

                        if (request.getInstallments() == null || request.getInstallments() <= 0) {
                                return ResponseEntity.badRequest().body(
                                                new SelectPlanResponseDto(false, null,
                                                                "Valid number of installments is required"));
                        }

                        logger.info("Processing select plan for session: {}, Amount: €{}, Installments: {}",
                                        request.getSessionId(), request.getDownPaymentAmount(),
                                        request.getInstallments());

                        // Retrieve session first
                        UUID sessionUuid;
                        try {
                                sessionUuid = UUID.fromString(request.getSessionId());
                        } catch (IllegalArgumentException e) {
                                logger.error("Invalid UUID format: {}", request.getSessionId());
                                return ResponseEntity.badRequest().body(
                                                new SelectPlanResponseDto(false, null, "Invalid Session ID format"));
                        }

                        Optional<PaymentSession> sessionOpt = paymentSessionRepository.findById(sessionUuid);
                        if (!sessionOpt.isPresent()) {
                                logger.warn("Session {} not found", request.getSessionId());
                                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(
                                                new SelectPlanResponseDto(false, null, "Session not found"));
                        }
                        PaymentSession session = sessionOpt.get();

                        // Check if plan already exists
                        if (installmentPlanRepository.findByPaymentSession(session).isPresent()) {
                                logger.warn("Installment plan already exists for session {}", request.getSessionId());
                                return ResponseEntity.badRequest().body(
                                                new SelectPlanResponseDto(false, null,
                                                                "Installment plan already exists for this session"));
                        }

                        // Create Installment Plan
                        InstallmentPlan plan = new InstallmentPlan();
                        plan.setInstallmentCount((long) request.getInstallments());
                        plan.setTotalAmount(session.getAmount());
                        plan.setPaymentSession(session);
                        plan.setInterestRate(0L);

                        long amountPerInstallment = session.getAmount() / request.getInstallments();
                        plan.setAmountPerInstallment(amountPerInstallment);

                        installmentPlanRepository.save(plan);

                        // Create Installments
                        List<Installment> installmentList = new ArrayList<>();
                        long remainder = session.getAmount() % request.getInstallments();

                        for (int i = 0; i < request.getInstallments(); i++) {
                                Installment installment = new Installment();
                                installment.setAmount(amountPerInstallment);
                                if (i == 0 && remainder > 0) {
                                        installment.setAmount(amountPerInstallment + remainder);
                                }
                                installment.setStatus("PENDING");
                                installment.setInstallmentPlan(plan);
                                installmentList.add(installment);
                        }

                        installmentsRepository.saveAll(installmentList);

                        // Initiate payment
                        String paymentId = plaidService.initiatePayment(request.getDownPaymentAmount());
                        logger.info("Payment initiated successfully. Payment ID: {}", paymentId);

                        // Mark first installment as completed
                        if (!installmentList.isEmpty()) {
                                Installment firstInstallment = installmentList.get(0);
                                firstInstallment.setStatus("COMPLETED");
                                installmentsRepository.save(firstInstallment);
                        }

                        // Update session status
                        session.setStatus("COMPLETED");
                        paymentSessionRepository.save(session);
                        logger.info("Session {} status updated to COMPLETED", request.getSessionId());

                        return ResponseEntity.ok(new SelectPlanResponseDto(true, paymentId,
                                        "Payment initiated, plan created, and session updated"));

                } catch (IOException e) {
                        logger.error("IOException while initiating payment: {}", e.getMessage(), e);
                        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(
                                        new SelectPlanResponseDto(false, null,
                                                        "Failed to initiate payment: " + e.getMessage()));
                } catch (Exception e) {
                        logger.error("Unexpected error in select-plan: {}", e.getMessage(), e);
                        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(
                                        new SelectPlanResponseDto(false, null,
                                                        "An unexpected error occurred: " + e.getMessage()));
                }
        }

        /**
         * Map Plaid transactions to TransactionDto objects
         *
         * @param transactions List of Plaid transactions
         * @return List of TransactionDto objects
         */
        private List<TransactionDto> mapTransactions(List<Transaction> transactions) {
                return transactions.stream()
                                .map(transaction -> new TransactionDto(
                                                transaction.getTransactionId(),
                                                transaction.getName(),
                                                transaction.getAmount(),
                                                transaction.getDate(),
                                                transaction.getPersonalFinanceCategory() != null
                                                                ? transaction.getPersonalFinanceCategory().getPrimary()
                                                                : "Uncategorized",
                                                transaction.getMerchantName(),
                                                transaction.getPending()))
                                .collect(Collectors.toList());
        }

        /**
         * Log all transactions with detailed information
         *
         * @param transactions List of transactions to log
         */
        private void logTransactions(List<TransactionDto> transactions) {
                logger.info("==================== TRANSACTION DETAILS ====================");
                logger.info("Total transactions: {}", transactions.size());

                if (transactions.isEmpty()) {
                        logger.info("No transactions found in the last 90 days");
                } else {
                        for (int i = 0; i < transactions.size(); i++) {
                                TransactionDto tx = transactions.get(i);
                                logger.info("Transaction #{}: {} | Amount: €{} | Date: {} | Merchant: {} | Category: {} | Pending: {}",
                                                i + 1,
                                                tx.getName(),
                                                tx.getAmount(),
                                                tx.getDate(),
                                                tx.getMerchantName() != null ? tx.getMerchantName() : "N/A",
                                                tx.getCategory(),
                                                tx.getPending());
                        }

                        // Summary statistics
                        double totalSpent = transactions.stream()
                                        .filter(tx -> tx.getAmount() > 0)
                                        .mapToDouble(TransactionDto::getAmount)
                                        .sum();

                        double totalIncome = transactions.stream()
                                        .filter(tx -> tx.getAmount() < 0)
                                        .mapToDouble(TransactionDto::getAmount)
                                        .sum();

                        logger.info("==================== TRANSACTION SUMMARY ====================");
                        logger.info("Total Spent: €{}", totalSpent);
                        logger.info("Total Income: €{}", Math.abs(totalIncome));
                        logger.info("Net: €{}", totalIncome + totalSpent);
                }
                logger.info("=============================================================");
        }

        /**
         * Generate installment plan options based on ML-approved max installments
         * Plans are offered in multiples of 4 up to the max allowed
         *
         * @param amount          Transaction amount
         * @param maxInstallments Maximum installments approved by ML (4-48)
         * @return List of installment plan options
         */
        private List<InstallmentPlanDto> generateInstallmentPlans(Double amount, int maxInstallments) {
                List<InstallmentPlanDto> plans = new ArrayList<>();

                // Always offer Pay in Full
                plans.add(new InstallmentPlanDto(
                                1,
                                amount,
                                "Pay in Full"));

                // Offer plans in multiples of 4 up to maxInstallments
                int[] installmentOptions = { 4, 8, 12, 16, 24, 36, 48 };

                for (int installments : installmentOptions) {
                        if (installments <= maxInstallments) {
                                long totalCents = Math.round(amount * 100);
                                long part = totalCents / installments;
                                long remainder = totalCents % installments;
                                double monthlyPayment = (part + remainder) / 100.0;
                                if (monthlyPayment >= 1.00) {
                                        plans.add(new InstallmentPlanDto(
                                                        installments,
                                                        monthlyPayment,
                                                        "Pay in " + installments));
                                }
                        }
                }

                return plans;
        }
}
