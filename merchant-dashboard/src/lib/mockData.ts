import type {
    PaymentPlan,
    Transaction,
    DashboardMetrics,
    RevenueDataPoint,
    StatusDistribution
} from "@/types"

// Mock Payment Plans
export const mockPaymentPlans: PaymentPlan[] = [
    {
        id: "plan_001",
        customerId: "cust_001",
        customerName: "Sarah Johnson",
        totalAmount: 1200,
        currency: "EUR",
        status: "Active",
        installments: [
            {
                id: "inst_001",
                planId: "plan_001",
                amount: 300,
                dueDate: new Date("2025-11-15"),
                paidDate: new Date("2025-11-14"),
                status: "Paid",
            },
            {
                id: "inst_002",
                planId: "plan_001",
                amount: 300,
                dueDate: new Date("2025-12-15"),
                status: "Pending",
            },
        ],
        createdAt: new Date("2025-10-01"),
        updatedAt: new Date("2025-11-14"),
    },
    // Add more mock data as needed
]

// Mock Transactions
export const mockTransactions: Transaction[] = [
    {
        id: "tx_001",
        customerId: "cust_001",
        customerName: "Alice Smith",
        amount: 120.0,
        currency: "EUR",
        status: "Success",
        timestamp: new Date(),
        paymentMethod: "Credit Card",
    },
    {
        id: "tx_002",
        customerId: "cust_002",
        customerName: "Bob Jones",
        amount: 45.5,
        currency: "EUR",
        status: "Pending",
        timestamp: new Date(Date.now() - 3600000),
        paymentMethod: "Bank Transfer",
    },
    // Add more transactions
]

// Mock Dashboard Metrics
export const mockMetrics: DashboardMetrics = {
    totalRevenue: 12450.0,
    pendingAmount: 3200.0,
    activePlansCount: 124,
    completionRate: 89,
    period: {
        start: new Date("2025-11-01"),
        end: new Date("2025-11-30"),
    },
}

// Mock Revenue Data
export const mockRevenueData: RevenueDataPoint[] = [
    { date: new Date("2025-01-01"), amount: 1200, label: "Jan" },
    { date: new Date("2025-02-01"), amount: 2100, label: "Feb" },
    { date: new Date("2025-03-01"), amount: 1800, label: "Mar" },
    { date: new Date("2025-04-01"), amount: 2400, label: "Apr" },
    { date: new Date("2025-05-01"), amount: 3200, label: "May" },
    { date: new Date("2025-06-01"), amount: 3800, label: "Jun" },
    { date: new Date("2025-07-01"), amount: 4200, label: "Jul" },
]

// Mock Status Distribution
export const mockStatusDistribution: StatusDistribution[] = [
    { status: "Completed", count: 45, percentage: 45 },
    { status: "Active", count: 30, percentage: 30 },
    { status: "Overdue", count: 15, percentage: 15 },
    { status: "Cancelled", count: 10, percentage: 10 },
]

// Helper function to format currency
export const formatCurrency = (amount: number, currency: string = "EUR"): string => {
    return new Intl.NumberFormat("en-EU", {
        style: "currency",
        currency: currency,
    }).format(amount)
}

// Helper function to format date
export const formatDate = (date: Date): string => {
    return new Intl.DateTimeFormat("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
    }).format(date)
}

// Helper function to calculate percentage change
export const calculatePercentageChange = (current: number, previous: number): number => {
    if (previous === 0) return 0
    return ((current - previous) / previous) * 100
}
