package com.morzio.server.mapper;

import org.springframework.stereotype.Component;

import com.morzio.server.dtos.PaymentService.PaymentServiceDto;
import com.morzio.server.entities.PaymentSession;

@Component
public class PaymentSessionMapper {
    
    /**
     * Maps a PaymentSession entity to PaymentServiceDto
     * @param paymentSession the entity to map
     * @return the mapped DTO
     */
    public PaymentServiceDto toDto(PaymentSession paymentSession) {
        if (paymentSession == null) {
            return null;
        }
        
        PaymentServiceDto dto = new PaymentServiceDto();
        dto.setId(paymentSession.getId());
        dto.setStatus(paymentSession.getStatus());
        dto.setAmount(paymentSession.getAmount());
        dto.setCreated(paymentSession.getCreated());
        
        return dto;
    }
    
    /**
     * Maps a PaymentServiceDto to PaymentSession entity
     * @param dto the DTO to map
     * @return the mapped entity
     */
    public PaymentSession toEntity(PaymentServiceDto dto) {
        if (dto == null) {
            return null;
        }
        
        PaymentSession entity = new PaymentSession();
        entity.setId(dto.getId());
        entity.setStatus(dto.getStatus());
        entity.setAmount(dto.getAmount());
        entity.setCreated(dto.getCreated());
        
        return entity;
    }
}
