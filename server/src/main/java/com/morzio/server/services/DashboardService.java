package com.morzio.server.services;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.morzio.server.dtos.Dashboard.*;
import com.morzio.server.repositorys.*;
import com.morzio.server.entities.*;

import java.util.List;
import java.util.ArrayList;
import java.util.stream.Collectors;

@Service
public class DashboardService {

        @Autowired
        private PaymentSessionRepository paymentSessionRepository;

        @Autowired
        private InstallmentsPlanRepository installmentPlanRepository;

        @Autowired
        private Installments installmentRepository;

        public DashboardMetricsDto getMetrics() {
                List<PaymentSession> allSessions = paymentSessionRepository.findAll();

                // Calculate total revenue (completed sessions)
                Double totalRevenue = allSessions.stream()
                                .filter(s -> "completed".equalsIgnoreCase(s.getStatus()))
                                .mapToDouble(s -> s.getAmount() / 100.0)
                                .sum();

                // Calculate pending amount
                Double pendingAmount = allSessions.stream()
                                .filter(s -> "pending".equalsIgnoreCase(s.getStatus()))
                                .mapToDouble(s -> s.getAmount() / 100.0)
                                .sum();

                // Get active plans count
                Integer activePlansCount = (int) installmentPlanRepository.count();

                // Calculate completion rate
                long completedCount = allSessions.stream()
                                .filter(s -> "completed".equalsIgnoreCase(s.getStatus()))
                                .count();
                Double completionRate = allSessions.isEmpty() ? 0.0 : (completedCount * 100.0) / allSessions.size();

                return new DashboardMetricsDto(
                                totalRevenue,
                                pendingAmount,
                                activePlansCount,
                                completionRate,
                                "+12%");
        }

        public List<RevenueDataPointDto> getRevenueData() {
                // For now, return mock data
                // In production, this would aggregate from database by month
                List<RevenueDataPointDto> data = new ArrayList<>();
                String[] months = { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul" };
                Double[] amounts = { 1200.0, 2100.0, 1800.0, 2400.0, 3200.0, 3800.0, 4200.0 };

                for (int i = 0; i < months.length; i++) {
                        data.add(new RevenueDataPointDto(months[i], amounts[i]));
                }

                return data;
        }

        public List<MonthlyInstallmentDto> getMonthlyInstallments() {
                // Mock data for monthly installments
                List<MonthlyInstallmentDto> data = new ArrayList<>();
                String[] months = { "Jan", "Feb", "Mar", "Apr", "May", "Jun" };
                Double[] amounts = { 4200.0, 3800.0, 5200.0, 4900.0, 6100.0, 5800.0 };

                for (int i = 0; i < months.length; i++) {
                        data.add(new MonthlyInstallmentDto(months[i], amounts[i]));
                }

                return data;
        }

        public List<StatusDistributionDto> getStatusDistribution() {
                List<PaymentSession> allSessions = paymentSessionRepository.findAll();
                long total = allSessions.size();

                if (total == 0) {
                        return new ArrayList<>();
                }

                long completed = allSessions.stream()
                                .filter(s -> "completed".equalsIgnoreCase(s.getStatus()))
                                .count();
                long active = allSessions.stream()
                                .filter(s -> "active".equalsIgnoreCase(s.getStatus()))
                                .count();
                long pending = allSessions.stream()
                                .filter(s -> "pending".equalsIgnoreCase(s.getStatus()))
                                .count();
                long failed = allSessions.stream()
                                .filter(s -> "failed".equalsIgnoreCase(s.getStatus()))
                                .count();

                List<StatusDistributionDto> distribution = new ArrayList<>();
                distribution.add(new StatusDistributionDto("Completed", (int) completed, (completed * 100.0) / total));
                distribution.add(new StatusDistributionDto("Active", (int) active, (active * 100.0) / total));
                distribution.add(new StatusDistributionDto("Pending", (int) pending, (pending * 100.0) / total));
                distribution.add(new StatusDistributionDto("Overdue", (int) failed, (failed * 100.0) / total));

                return distribution;
        }

        public List<RecentTransactionDto> getRecentTransactions(int limit, String status) {
                List<PaymentSession> sessions = paymentSessionRepository.findAll();

                return sessions.stream()
                                .filter(session -> status == null || status.isEmpty() ||
                                                session.getStatus().equalsIgnoreCase(status))
                                .sorted((a, b) -> b.getCreated().compareTo(a.getCreated()))
                                .limit(limit)
                                .map(session -> new RecentTransactionDto(
                                                session.getId().toString(),
                                                "Customer-" + session.getId().toString().substring(0, 8),
                                                session.getAmount() / 100.0,
                                                session.getStatus(),
                                                session.getCreated()))
                                .collect(Collectors.toList());
        }

        public List<PaymentPlanSummaryDto> getPaymentPlans(String status, int limit) {
                List<InstallmentPlan> plans = installmentPlanRepository.findAll();

                return plans.stream()
                                .limit(limit)
                                .map(plan -> {
                                        // Get paid installments count
                                        List<Installment> installments = installmentRepository
                                                        .findByInstallmentPlanId(plan.getId());

                                        int paidCount = (int) installments.stream()
                                                        .filter(i -> "paid".equalsIgnoreCase(i.getStatus()))
                                                        .count();

                                        return new PaymentPlanSummaryDto(
                                                        plan.getId().toString(),
                                                        "Customer-" + plan.getId(),
                                                        plan.getTotalAmount() / 100.0,
                                                        plan.getInstallmentCount().intValue(),
                                                        paidCount,
                                                        "Dec 15, 2025",
                                                        paidCount == plan.getInstallmentCount() ? "Completed"
                                                                        : "Active");
                                })
                                .collect(Collectors.toList());
        }
}
