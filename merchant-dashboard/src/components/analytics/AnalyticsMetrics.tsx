"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { motion } from "framer-motion"
import { TrendingUp, Users, CreditCard, Activity } from "lucide-react"

const metrics = [
    {
        title: "Avg. Transaction",
        value: "€342.50",
        change: "+12.5%",
        trend: "up",
        icon: CreditCard,
        color: "text-blue-600",
        bg: "bg-blue-100"
    },
    {
        title: "Conversion Rate",
        value: "3.2%",
        change: "+0.4%",
        trend: "up",
        icon: Activity,
        color: "text-green-600",
        bg: "bg-green-100"
    },
    {
        title: "Total Customers",
        value: "1,294",
        change: "+18.2%",
        trend: "up",
        icon: Users,
        color: "text-purple-600",
        bg: "bg-purple-100"
    },
    {
        title: "Growth Rate",
        value: "24.5%",
        change: "-2.1%",
        trend: "down",
        icon: TrendingUp,
        color: "text-amber-600",
        bg: "bg-amber-100"
    }
]

export function AnalyticsMetrics() {
    return (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {metrics.map((metric, index) => (
                <motion.div
                    key={metric.title}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                >
                    <Card className="hover:shadow-lg transition-all duration-200 border-none shadow-sm bg-white/50 backdrop-blur-sm">
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium text-muted-foreground">
                                {metric.title}
                            </CardTitle>
                            <div className={`p-2 rounded-full ${metric.bg}`}>
                                <metric.icon className={`h-4 w-4 ${metric.color}`} />
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">{metric.value}</div>
                            <p className={`text-xs mt-1 font-medium ${metric.trend === 'up' ? 'text-green-600' : 'text-red-600'
                                }`}>
                                {metric.change} <span className="text-muted-foreground font-normal">from last month</span>
                            </p>
                        </CardContent>
                    </Card>
                </motion.div>
            ))}
        </div>
    )
}
