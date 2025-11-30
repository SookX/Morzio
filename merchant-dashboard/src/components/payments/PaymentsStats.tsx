"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { motion } from "framer-motion"
import { DollarSign, TrendingUp, TrendingDown, Activity } from "lucide-react"

const stats = [
    {
        title: "Total Volume",
        value: "€45,231.89",
        change: "+20.1% from last month",
        trend: "up",
        icon: DollarSign,
    },
    {
        title: "Success Rate",
        value: "98.5%",
        change: "+2.5% from last month",
        trend: "up",
        icon: Activity,
    },
    {
        title: "Avg. Transaction",
        value: "€156.45",
        change: "-4.2% from last month",
        trend: "down",
        icon: TrendingDown,
    },
    {
        title: "Total Transactions",
        value: "289",
        change: "+12.3% from last month",
        trend: "up",
        icon: TrendingUp,
    },
]

export function PaymentsStats() {
    return (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {stats.map((stat, index) => (
                <motion.div
                    key={stat.title}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                >
                    <Card className="hover:shadow-md transition-shadow duration-200">
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium text-muted-foreground">
                                {stat.title}
                            </CardTitle>
                            <stat.icon className={`h-4 w-4 ${stat.trend === "up" ? "text-green-600" : "text-red-600"
                                }`} />
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">{stat.value}</div>
                            <p className={`text-xs mt-1 ${stat.trend === "up" ? "text-green-600" : "text-red-600"
                                }`}>
                                {stat.change}
                            </p>
                        </CardContent>
                    </Card>
                </motion.div>
            ))}
        </div>
    )
}
