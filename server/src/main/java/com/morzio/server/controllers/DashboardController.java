package com.morzio.server.controllers;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import com.morzio.server.dtos.Dashboard.*;
import com.morzio.server.services.DashboardService;

import java.util.List;

@RestController
@RequestMapping("/api/dashboard")
public class DashboardController {

    @Autowired
    private DashboardService dashboardService;

    @GetMapping("/metrics")
    public DashboardMetricsDto getMetrics() {
        return dashboardService.getMetrics();
    }

    @GetMapping("/revenue")
    public List<RevenueDataPointDto> getRevenueData() {
        return dashboardService.getRevenueData();
    }

    @GetMapping("/installments")
    public List<MonthlyInstallmentDto> getMonthlyInstallments() {
        return dashboardService.getMonthlyInstallments();
    }

    @GetMapping("/status-distribution")
    public List<StatusDistributionDto> getStatusDistribution() {
        return dashboardService.getStatusDistribution();
    }

    @GetMapping("/recent-transactions")
    public List<RecentTransactionDto> getRecentTransactions(
            @RequestParam(defaultValue = "10") int limit,
            @RequestParam(required = false) String status) {
        return dashboardService.getRecentTransactions(limit, status);
    }

    @GetMapping("/payment-plans")
    public List<PaymentPlanSummaryDto> getPaymentPlans(
            @RequestParam(required = false) String status,
            @RequestParam(defaultValue = "10") int limit) {
        return dashboardService.getPaymentPlans(status, limit);
    }
}
