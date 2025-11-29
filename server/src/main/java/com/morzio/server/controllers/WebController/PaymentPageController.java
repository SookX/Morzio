package com.morzio.server.controllers.WebController;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import com.morzio.server.services.PaymentService.PaymentServiceImpl.PaymentServiceImpl;
import com.morzio.server.dtos.PaymentService.PaymentServiceDto;
import java.util.UUID;

@Controller
public class PaymentPageController {

    @Autowired
    PaymentServiceImpl paymentService;

    @GetMapping("/pay/{sessionId}")
    public String paymentPage(@PathVariable String sessionId, Model model) {
        try {
            UUID uuid = UUID.fromString(sessionId);
            PaymentServiceDto payment = paymentService.getPaymentSession(uuid);
            double amountFormatted = payment.getAmount() / 100.0;
            model.addAttribute("amount", String.format("%.2f", amountFormatted));
            model.addAttribute("sessionId", sessionId);
            return "payment";
        } catch (Exception e) {
            return "error";
        }
    }
}
