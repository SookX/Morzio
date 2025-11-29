package com.morzio.server.controllers.PaymentsController;

import com.morzio.server.dtos.PaymentService.PaymentServiceDto;
import com.morzio.server.services.PaymentService.PaymentService;
import com.morzio.server.services.PlaidService;
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

    public PaymentPageController(PaymentService paymentService, PlaidService plaidService) {
        this.paymentService = paymentService;
        this.plaidService = plaidService;
    }

    @GetMapping("/pay/{sessionId}")
    public String paymentPage(@PathVariable String sessionId, Model model) throws IOException {
        PaymentServiceDto session = paymentService.getPaymentSession(UUID.fromString(sessionId));

        // Use session ID as client user ID for now
        String linkToken = plaidService.createLinkToken(session.getId().toString());

        model.addAttribute("amount", session.getAmount());
        model.addAttribute("linkToken", linkToken);

        return "payment";
    }
}
