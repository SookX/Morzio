package com.morzio.server.controllers;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class PageController {

    @GetMapping("/success")
    public String successPage() {
        return "success";
    }

    @GetMapping("/error")
    public String errorPage() {
        return "error";
    }

    @GetMapping("/")
    public String landingPage(@RequestParam(name = "status", required = false) String status, Model model) {
        // If the user returns after a payment, we can show a specific welcome back message
        if ("success".equals(status)) {
            model.addAttribute("paymentMessage", "Payment successful! Welcome back to Morzio.");
        }
        return "lander";
    }
}
