package com.morzio.server.services.PaymentService.PaymentServiceImpl;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.morzio.server.dtos.PaymentService.PaymentInitiateResponseDto;
import com.morzio.server.dtos.PaymentService.PaymentServiceDto;
import com.morzio.server.entities.PaymentSession;
import com.morzio.server.mapper.PaymentSessionMapper;
import com.morzio.server.repositorys.PaymentSessionRepository;
import com.morzio.server.services.PaymentService.PaymentService;

@Service
public class PaymentServiceImpl implements PaymentService{
    
    @Autowired
    PaymentSessionRepository paymentSessionRepository;
    
    @Autowired
    PaymentSessionMapper paymentSessionMapper;

    @Override
    public PaymentInitiateResponseDto url_for_qr(Long amount) {

        PaymentServiceDto paymentServiceDto = createPaymentService(amount);

        String baseUrl = "http://morzio.com/pay/";
        String urlWithSessionId = baseUrl + paymentServiceDto.getId().toString();

        PaymentInitiateResponseDto response = new PaymentInitiateResponseDto();
        response.setSessionId(paymentServiceDto.getId());
        response.setStatus(paymentServiceDto.getStatus());
        response.setPaymentUrl(urlWithSessionId);

        return response;
    }

    public PaymentServiceDto createPaymentService(Long amount) {
        PaymentSession paymentSession = new PaymentSession();
        paymentSession.setStatus("PENDING");
        paymentSession.setAmount(amount);
        
        PaymentSession savedSession = paymentSessionRepository.save(paymentSession);
        
        PaymentServiceDto dto = paymentSessionMapper.toDto(savedSession);
        
        return dto;
    }}
