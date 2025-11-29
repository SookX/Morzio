package com.morzio.server.controllers.PaymentsController;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.morzio.server.dtos.PaymentService.PaymentInitiateRequestDto;
import com.morzio.server.dtos.PaymentService.PaymentInitiateResponseDto;
import com.morzio.server.services.PaymentService.PaymentServiceImpl.PaymentServiceImpl;

@RestController
@RequestMapping("/api")
public class PaymentController {
    
    @Autowired
    PaymentServiceImpl paymentService;
    
    @GetMapping("/ping")
    public String ping() {
        return "pong";
    }
    
    /**
     * @param request
     * @return url for the qr
     */
    @PostMapping("/payment/initiate")
    public PaymentInitiateResponseDto initiatePayment(@RequestBody PaymentInitiateRequestDto request) {
        PaymentInitiateResponseDto response =  paymentService.url_for_qr(request.getAmount());
        
        return response;
    }
}
