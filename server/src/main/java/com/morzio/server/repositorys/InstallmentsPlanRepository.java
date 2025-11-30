package com.morzio.server.repositorys;

import org.springframework.data.jpa.repository.JpaRepository;
import com.morzio.server.entities.InstallmentPlan;
import com.morzio.server.entities.PaymentSession;
import java.util.Optional;
import java.util.UUID;

public interface InstallmentsPlanRepository extends JpaRepository<InstallmentPlan, Long> {
    Optional<InstallmentPlan> findByPaymentSession(PaymentSession paymentSession);

    Optional<InstallmentPlan> findByPaymentSessionId(UUID sessionId);
}
