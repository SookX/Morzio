"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip, Legend } from "recharts"
import { motion } from "framer-motion"
import { useEffect, useState } from "react"
import { dashboardAPI } from "@/lib/api"

interface StatusData {
    status: string
    count: number
    percentage: number
    [key: string]: any
}

const COLORS = {
    Completed: "#10B981",
    Active: "#4F46E5",
    Pending: "#F59E0B",
    Overdue: "#EF4444",
    Cancelled: "#6B7280"
}

export function PaymentStatusChart() {
    const [data, setData] = useState<StatusData[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        async function loadData() {
            try {
                const response = await dashboardAPI.getStatusDistribution()
                setData(response as unknown as StatusData[])
            } catch (error) {
                console.error("Failed to load status data:", error)
            } finally {
                setLoading(false)
            }
        }
        loadData()
    }, [])

    if (loading) {
        return (
            <Card className="col-span-4 lg:col-span-3 h-full">
                <CardHeader>
                    <CardTitle>Payment Status Distribution</CardTitle>
                </CardHeader>
                <CardContent className="h-[300px] flex items-center justify-center">
                    <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
                </CardContent>
            </Card>
        )
    }

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="col-span-4 lg:col-span-3"
        >
            <Card className="h-full">
                <CardHeader>
                    <CardTitle>Payment Status Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={data}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={90}
                                    paddingAngle={5}
                                    dataKey="count"
                                    nameKey="status"
                                >
                                    {data.map((entry, index) => (
                                        <Cell
                                            key={`cell-${index}`}
                                            fill={COLORS[entry.status as keyof typeof COLORS] || "#CBD5E1"}
                                        />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: "#fff",
                                        borderRadius: "8px",
                                        border: "1px solid #E5E7EB",
                                        boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)"
                                    }}
                                />
                                <Legend
                                    verticalAlign="bottom"
                                    height={36}
                                    iconType="circle"
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </CardContent>
            </Card>
        </motion.div>
    )
}
