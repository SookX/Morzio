"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from "recharts"
import { motion } from "framer-motion"
import { useEffect, useState } from "react"
import { dashboardAPI } from "@/lib/api"

interface RevenueData {
    month: string
    total: number
}

export function RevenueChart() {
    const [data, setData] = useState<RevenueData[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        async function loadData() {
            try {
                const response = await dashboardAPI.getRevenueData()
                setData(response as unknown as RevenueData[])
            } catch (error) {
                console.error("Failed to load revenue data:", error)
            } finally {
                setLoading(false)
            }
        }
        loadData()
    }, [])

    if (loading) {
        return (
            <Card className="col-span-4 h-full">
                <CardHeader>
                    <CardTitle>Revenue Overview</CardTitle>
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
            transition={{ duration: 0.5, delay: 0.2 }}
            className="col-span-4"
        >
            <Card className="h-full">
                <CardHeader>
                    <CardTitle>Revenue Overview</CardTitle>
                </CardHeader>
                <CardContent className="pl-2">
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                                <XAxis
                                    dataKey="month"
                                    stroke="#888888"
                                    fontSize={12}
                                    tickLine={false}
                                    axisLine={false}
                                    tickMargin={10}
                                />
                                <YAxis
                                    stroke="#888888"
                                    fontSize={12}
                                    tickLine={false}
                                    axisLine={false}
                                    tickFormatter={(value) => `€${value}`}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: "#fff",
                                        borderRadius: "8px",
                                        border: "1px solid #E5E7EB",
                                        boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)"
                                    }}
                                    cursor={{ stroke: "#E5E7EB", strokeWidth: 2 }}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="total"
                                    stroke="#4F46E5"
                                    strokeWidth={3}
                                    dot={{ r: 4, fill: "#4F46E5", strokeWidth: 2, stroke: "#fff" }}
                                    activeDot={{ r: 6, strokeWidth: 0 }}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </CardContent>
            </Card>
        </motion.div>
    )
}
