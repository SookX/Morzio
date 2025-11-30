"use client"

import { AnalyticsMetrics } from "@/components/analytics/AnalyticsMetrics"
import { DetailedRevenueChart } from "@/components/analytics/DetailedRevenueChart"
import { CustomerGrowthChart } from "@/components/analytics/CustomerGrowthChart"
import { PaymentMethodChart } from "@/components/analytics/PaymentMethodChart"
import { Button } from "@/components/ui/button"
import { Calendar as CalendarIcon, Download } from "lucide-react"

export default function AnalyticsPage() {
    return (
        <div className="space-y-8">
            {/* Header Section */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Analytics</h1>
                    <p className="text-muted-foreground mt-1">
                        Deep dive into your payment analytics and trends.
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <Button variant="outline" className="hidden sm:flex">
                        <CalendarIcon className="mr-2 h-4 w-4" />
                        Last 30 Days
                    </Button>
                    <Button variant="outline">
                        <Download className="mr-2 h-4 w-4" />
                        Export
                    </Button>
                </div>
            </div>

            {/* Key Metrics */}
            <AnalyticsMetrics />

            {/* Main Charts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                {/* Revenue Trends - Full Width on Mobile, 4 cols on Desktop */}
                <DetailedRevenueChart />

                {/* Secondary Charts - 2 cols each on Desktop */}
                <CustomerGrowthChart />
                <PaymentMethodChart />
            </div>
        </div>
    )
}
