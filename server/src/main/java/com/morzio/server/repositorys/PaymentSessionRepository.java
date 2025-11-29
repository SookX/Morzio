package com.morzio.server.repositorys;

import org.springframework.data.jpa.repository.JpaRepository;

import com.morzio.server.entities.PaymentSession;

public interface PaymentSessionRepository extends JpaRepository<PaymentSession, Long> {

}