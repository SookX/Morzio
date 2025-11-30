"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { motion } from "framer-motion"
import { DollarSign, CreditCard, Clock, CheckCircle } from "lucide-react"
import { useEffect, useState } from "react"
import { dashboardAPI } from "@/lib/api"
import { formatCurrency } from "@/lib/mockData"

interface MetricsState {
    totalRevenue: number
    pendingAmount: number
    activePlansCount: number
    completionRate: number
    changePercentage: string
}

export function MetricsCards() {
    const [metrics, setMetrics] = useState<MetricsState | null>(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        async function loadMetrics() {
            try {
                const data = await dashboardAPI.getMetrics()
                setMetrics(data as unknown as MetricsState)
            } catch (error) {
                console.error("Failed to load metrics:", error)
            } finally {
                setLoading(false)
            }
        }
        loadMetrics()
    }, [])

    const cards = [
        {
            title: "Total Revenue",
            value: metrics ? formatCurrency(metrics.totalRevenue) : "...",
            change: metrics ? metrics.changePercentage : "...",
            icon: DollarSign,
            color: "text-green-600",
        },
        {
            title: "Pending Installments",
            value: metrics ? formatCurrency(metrics.pendingAmount) : "...",
            change: "Pending collection",
            icon: Clock,
            color: "text-amber-500",
        },
        {
            title: "Active Plans",
            value: metrics ? metrics.activePlansCount.toString() : "...",
            change: "Currently active",
            icon: CreditCard,
            color: "text-blue-500",
        },
        {
            title: "Completion Rate",
            value: metrics ? `${Math.round(metrics.completionRate)}%` : "...",
            change: "Overall performance",
            icon: CheckCircle,
            color: "text-indigo-500",
        },
    ]

    return (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {cards.map((metric, index) => (
                <motion.div
                    key={metric.title}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                >
                    <Card className="hover:shadow-md transition-shadow duration-200">
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium text-muted-foreground">
                                {metric.title}
                            </CardTitle>
                            <metric.icon className={`h-4 w-4 ${metric.color}`} />
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">
                                {loading ? (
                                    <div className="h-8 w-24 bg-gray-200 animate-pulse rounded" />
                                ) : (
                                    metric.value
                                )}
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                                {metric.change}
                            </p>
                        </CardContent>
                    </Card>
                </motion.div>
            ))}
        </div>
    )
}
