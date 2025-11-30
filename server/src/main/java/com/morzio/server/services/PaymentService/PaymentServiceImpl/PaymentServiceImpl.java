package com.morzio.server.services.PaymentService.PaymentServiceImpl;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.morzio.server.dtos.PaymentService.PaymentInitiateResponseDto;
import com.morzio.server.dtos.PaymentService.PaymentServiceDto;
import com.morzio.server.entities.PaymentSession;
import com.morzio.server.mapper.PaymentSessionMapper;
import com.morzio.server.repositorys.PaymentSessionRepository;
import com.morzio.server.services.PaymentService.PaymentService;
import com.morzio.server.repositorys.InstallmentsPlanRepository;
import com.morzio.server.repositorys.Installments;
import com.morzio.server.entities.InstallmentPlan;
import com.morzio.server.entities.Installment;
import com.morzio.server.dtos.PaymentService.InstallmentDto;
import java.util.List;
import java.util.stream.Collectors;
import java.util.Optional;

@Service
public class PaymentServiceImpl implements PaymentService {

    @Autowired
    PaymentSessionRepository paymentSessionRepository;

    @Autowired
    PaymentSessionMapper paymentSessionMapper;

    @Autowired
    InstallmentsPlanRepository installmentsPlanRepository;

    @Autowired
    Installments installmentsRepository;

    @Value("${app.base-url:http://localhost:8080}")
    private String appBaseUrl;

    @Override
    public PaymentInitiateResponseDto url_for_qr(Long amount) {

        PaymentServiceDto paymentServiceDto = createPaymentService(amount);

        String baseUrl = appBaseUrl + "/pay/";
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
    }

    @Override
    public PaymentServiceDto getPaymentSession(java.util.UUID id) {
        PaymentSession session = paymentSessionRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Payment session not found"));

        PaymentServiceDto dto = paymentSessionMapper.toDto(session);

        Optional<InstallmentPlan> plan = installmentsPlanRepository.findByPaymentSessionId(id);
        if (plan.isPresent()) {
            List<Installment> installmentEntities = installmentsRepository.findByInstallmentPlanId(plan.get().getId());
            List<InstallmentDto> installmentDtos = new java.util.ArrayList<>();
            java.time.LocalDate startDate = java.time.LocalDate.now().plusMonths(1);

            for (int i = 0; i < installmentEntities.size(); i++) {
                Installment installment = installmentEntities.get(i);
                String dueDate = startDate.plusMonths(i).toString();
                installmentDtos.add(new InstallmentDto(installment.getAmount(), installment.getStatus(), dueDate));
            }
            dto.setInstallments(installmentDtos);
        }

        return dto;
    }
}
