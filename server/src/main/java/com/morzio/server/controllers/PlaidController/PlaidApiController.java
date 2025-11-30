package com.morzio.server.controllers.PlaidController;

import com.morzio.server.dtos.PlaidService.InstallmentPlanDto;
import com.morzio.server.dtos.PlaidService.PlaidOnSuccessRequestDto;
import com.morzio.server.dtos.PlaidService.PlaidOnSuccessResponseDto;
import com.morzio.server.dtos.PlaidService.TransactionDto;
import com.morzio.server.services.PlaidService;
import com.plaid.client.model.AccountBase;
import com.plaid.client.model.AccountsGetResponse;
import com.plaid.client.model.Transaction;
import com.plaid.client.model.TransactionsGetResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/plaid")
public class PlaidApiController {

    private static final Logger logger = LoggerFactory.getLogger(PlaidApiController.class);

    @Autowired
    private PlaidService plaidService;

    /**
     * Handles the Plaid Link success callback
     * Exchanges public token, fetches account balances and transactions,
     * and returns installment plan options based on risk logic
     *
     * @param request Contains publicToken and transactionAmount
     * @return PlaidOnSuccessResponseDto with installment plans or error
     */
    @PostMapping("/on-success")
    public ResponseEntity<PlaidOnSuccessResponseDto> handlePlaidSuccess(@RequestBody PlaidOnSuccessRequestDto request) {
        try {
            // Validate request
            if (request.getPublicToken() == null || request.getPublicToken().isEmpty()) {
                return ResponseEntity.badRequest().body(
                        new PlaidOnSuccessResponseDto(false, null, null, null, "Public token is required"));
            }

            if (request.getTransactionAmount() == null || request.getTransactionAmount() <= 0) {
                return ResponseEntity.badRequest().body(
                        new PlaidOnSuccessResponseDto(false, null, null, null, "Valid transaction amount is required"));
            }

            logger.info("Processing Plaid success callback for transaction amount: £{}",
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

            // Step 4: Mock risk logic - check if account balance > transaction amount
            Double totalBalance = calculateTotalBalance(accountsResponse.getAccounts());
            Double transactionAmount = request.getTransactionAmount();

            logger.info("Total available balance: £{}, Transaction amount: £{}", totalBalance, transactionAmount);

            if (totalBalance >= transactionAmount) {
                // User has sufficient balance - return installment plan options
                List<InstallmentPlanDto> installmentPlans = generateInstallmentPlans(transactionAmount);

                logger.info("Risk check PASSED - Offering {} installment plan options", installmentPlans.size());

                return ResponseEntity.ok(
                        new PlaidOnSuccessResponseDto(
                                true,
                                "Installment plans available",
                                installmentPlans,
                                transactionDtos,
                                null));
            } else {
                // Insufficient balance
                logger.warn("Risk check FAILED - Insufficient balance. Available: £{}, Required: £{}",
                        totalBalance, transactionAmount);

                return ResponseEntity.status(HttpStatus.PAYMENT_REQUIRED).body(
                        new PlaidOnSuccessResponseDto(
                                false,
                                null,
                                null,
                                transactionDtos,
                                String.format("Insufficient balance. Available: £%.2f, Required: £%.2f",
                                        totalBalance, transactionAmount)));
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
                logger.info("Transaction #{}: {} | Amount: £{} | Date: {} | Merchant: {} | Category: {} | Pending: {}",
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
            logger.info("Total Spent: £{}", totalSpent);
            logger.info("Total Income: £{}", Math.abs(totalIncome));
            logger.info("Net: £{}", totalIncome + totalSpent);
        }
        logger.info("=============================================================");
    }

    /**
     * Calculate total available balance across all accounts
     *
     * @param accounts List of account balances from Plaid
     * @return Total available balance
     */
    private Double calculateTotalBalance(List<AccountBase> accounts) {
        return accounts.stream()
                .filter(account -> account.getBalances() != null && account.getBalances().getAvailable() != null)
                .mapToDouble(account -> account.getBalances().getAvailable())
                .sum();
    }

    /**
     * Generate installment plan options: Pay in 1, Pay in 3, Pay in 6
     *
     * @param amount Transaction amount
     * @return List of installment plan options
     */
    private List<InstallmentPlanDto> generateInstallmentPlans(Double amount) {
        List<InstallmentPlanDto> plans = new ArrayList<>();

        // Pay in 1 (full amount)
        plans.add(new InstallmentPlanDto(
                1,
                amount,
                "Pay in 1"));

        // Pay in 3
        plans.add(new InstallmentPlanDto(
                3,
                Math.round(amount / 3 * 100.0) / 100.0,
                "Pay in 3"));

        // Pay in 6
        plans.add(new InstallmentPlanDto(
                6,
                Math.round(amount / 6 * 100.0) / 100.0,
                "Pay in 6"));

        return plans;
    }
}
