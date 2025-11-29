package com.morzio.server.repositorys;

import org.springframework.data.jpa.repository.JpaRepository;
import com.morzio.server.entities.Installment;

public interface Installments extends JpaRepository<Installment, Long>{
}