package com.morzio.server.repositorys;

import org.springframework.data.jpa.repository.JpaRepository;
import com.morzio.server.entities.Installment;
import java.util.List;

public interface Installments extends JpaRepository<Installment, Long> {
    List<Installment> findByInstallmentPlanId(Long installmentPlanId);
}