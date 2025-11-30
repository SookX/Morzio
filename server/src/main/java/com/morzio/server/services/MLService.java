package com.morzio.server.services;

import com.morzio.server.dtos.PlaidService.MLRequestDto;
import com.morzio.server.dtos.PlaidService.MLResponseDto;
import com.morzio.server.dtos.PlaidService.MLTransactionDto;
import com.morzio.server.dtos.PlaidService.TransactionDto;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class MLService {

    private static final Logger logger = LoggerFactory.getLogger(MLService.class);

    @Value("${ml.service.url:http://localhost:8000}")
    private String mlServiceUrl;

    private final RestTemplate restTemplate;

    public MLService() {
        this.restTemplate = new RestTemplate();
    }

    /**
     * Call the ML service to get risk assessment and max installments
     *
     * @param transactions      List of Plaid transactions
     * @param transactionAmount Current purchase amount
     * @return MLResponseDto with approved status and max installments
     */
    public MLResponseDto predict(List<TransactionDto> transactions, Double transactionAmount) {
        try {
            // Convert Plaid transactions to ML format
            List<MLTransactionDto> mlTransactions = transactions.stream()
                    .map(tx -> new MLTransactionDto(
                            tx.getAmount(),
                            tx.getDate(),
                            tx.getCategory()))
                    .collect(Collectors.toList());

            // Create request
            MLRequestDto request = new MLRequestDto();
            request.setTransactions(mlTransactions);
            request.setTransaction_amount(transactionAmount);
            request.setTransaction_mcc(5411); // Default grocery store MCC

            // Set headers
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<MLRequestDto> entity = new HttpEntity<>(request, headers);

            String url = mlServiceUrl + "/api/predict";
            logger.info("Calling ML service at: {}", url);
            logger.info("Request: {} transactions, amount: £{}", mlTransactions.size(), transactionAmount);

            // Call ML service
            MLResponseDto response = restTemplate.postForObject(url, entity, MLResponseDto.class);

            if (response != null) {
                logger.info("ML Response: approved={}, max_installments={}",
                        response.isApproved(), response.getMax_installments());
                return response;
            } else {
                logger.error("ML service returned null response");
                return new MLResponseDto(false, 0);
            }

        } catch (Exception e) {
            logger.error("Error calling ML service: {}", e.getMessage(), e);
            // Return declined on error for safety
            return new MLResponseDto(false, 0);
        }
    }

    /**
     * Check if ML service is healthy
     *
     * @return true if ML service is available
     */
    public boolean isHealthy() {
        try {
            String url = mlServiceUrl + "/health";
            String response = restTemplate.getForObject(url, String.class);
            return response != null && response.contains("healthy");
        } catch (Exception e) {
            logger.warn("ML service health check failed: {}", e.getMessage());
            return false;
        }
    }
}

