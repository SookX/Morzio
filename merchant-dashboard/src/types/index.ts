// Payment Plan Types
export interface PaymentPlan {
    id: string
    customerId: string
    customerName: string
    totalAmount: number
    currency: string
    installments: Installment[]
    status: PaymentPlanStatus
    createdAt: Date
    updatedAt: Date
}

export interface Installment {
    id: string
    planId: string
    amount: number
    dueDate: Date
    paidDate?: Date
    status: InstallmentStatus
}

export type PaymentPlanStatus = "Active" | "Completed" | "Overdue" | "Cancelled"
export type InstallmentStatus = "Pending" | "Paid" | "Overdue" | "Failed"

// Transaction Types
export interface Transaction {
    id: string
    customerId: string
    customerName: string
    amount: number
    currency: string
    status: TransactionStatus
    timestamp: Date
    paymentMethod?: string
}

export type TransactionStatus = "Success" | "Pending" | "Failed" | "Refunded"

// Metrics Types
export interface DashboardMetrics {
    totalRevenue: number
    pendingAmount: number
    activePlansCount: number
    completionRate: number
    period: {
        start: Date
        end: Date
    }
}

// Chart Data Types
export interface RevenueDataPoint {
    date: Date
    amount: number
    label: string
}

export interface StatusDistribution {
    status: PaymentPlanStatus
    count: number
    percentage: number
}

// API Response Types
export interface ApiResponse<T> {
    data: T
    success: boolean
    message?: string
    error?: string
}

export interface PaginatedResponse<T> {
    data: T[]
    total: number
    page: number
    pageSize: number
    hasMore: boolean
}

// Filter Types
export interface DateRangeFilter {
    startDate: Date
    endDate: Date
}

export interface PaymentPlanFilter extends DateRangeFilter {
    status?: PaymentPlanStatus[]
    customerId?: string
    minAmount?: number
    maxAmount?: number
}
