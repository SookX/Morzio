// API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8080'

// Generic fetch wrapper
async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
    try {
        console.log(`Fetching: ${API_BASE_URL}${endpoint}`)
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options?.headers,
            },
            ...options,
        })

        console.log(`Response status: ${response.status}`)

        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`)
        }

        return response.json()
    } catch (error) {
        console.error(`API Fetch Error for ${endpoint}:`, error)
        throw error
    }
}

// Dashboard API
export const dashboardAPI = {
    getMetrics: () => fetchAPI('/api/dashboard/metrics'),
    getRevenueData: () => fetchAPI('/api/dashboard/revenue'),
    getMonthlyInstallments: () => fetchAPI('/api/dashboard/installments'),
    getStatusDistribution: () => fetchAPI('/api/dashboard/status-distribution'),
    getRecentTransactions: (limit = 10, status?: string) => {
        const params = new URLSearchParams({ limit: limit.toString() })
        if (status && status !== 'all') params.append('status', status)
        return fetchAPI(`/api/dashboard/recent-transactions?${params}`)
    },
    getPaymentPlans: (status?: string, limit = 10) => {
        const params = new URLSearchParams({ limit: limit.toString() })
        if (status) params.append('status', status)
        return fetchAPI(`/api/dashboard/payment-plans?${params}`)
    },
}

// Payment API (for future use)
export const paymentAPI = {
    initiatePayment: (amount: number) =>
        fetchAPI('/api/payment/initiate', {
            method: 'POST',
            body: JSON.stringify({ amount }),
        }),
}
