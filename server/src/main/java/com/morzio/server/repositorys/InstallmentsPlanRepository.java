package com.morzio.server.repositorys;

import org.springframework.data.jpa.repository.JpaRepository;

import com.morzio.server.entities.InstallmentPlan;

public interface InstallmentsPlanRepository extends JpaRepository<InstallmentPlan, Long>{
    
}
