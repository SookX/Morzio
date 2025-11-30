package com.morzio.server.controllers.PaymentsController;

import com.morzio.server.dtos.PaymentService.PaymentServiceDto;
import com.morzio.server.services.PaymentService.PaymentService;
import com.morzio.server.services.PlaidService;
import com.morzio.server.repositorys.InstallmentsPlanRepository;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;

import java.io.IOException;
import java.util.UUID;

@Controller
public class PaymentPageController {

    private final PaymentService paymentService;
    private final PlaidService plaidService;
    private final InstallmentsPlanRepository installmentsPlanRepository;

    public PaymentPageController(PaymentService paymentService, PlaidService plaidService,
            InstallmentsPlanRepository installmentsPlanRepository) {
        this.paymentService = paymentService;
        this.plaidService = plaidService;
        this.installmentsPlanRepository = installmentsPlanRepository;
    }

    @GetMapping("/pay/{sessionId}")
    public String paymentPage(@PathVariable String sessionId, Model model) throws IOException {
        UUID sessionUuid = UUID.fromString(sessionId);

        // Check if installment plan already exists
        if (installmentsPlanRepository.findByPaymentSessionId(sessionUuid).isPresent()) {
            return "redirect:/error";
        }

        PaymentServiceDto session = paymentService.getPaymentSession(sessionUuid);

        String linkToken = plaidService.createLinkToken(session.getId().toString());

        double amountInDollars = session.getAmount() / 100.0;
        String formattedAmount = String.format("%.2f", amountInDollars);

        model.addAttribute("amount", formattedAmount);
        model.addAttribute("linkToken", linkToken);
        model.addAttribute("sessionId", sessionId);

        return "payment";
    }
}
