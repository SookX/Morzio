"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from "recharts"
import { motion } from "framer-motion"
import { useEffect, useState } from "react"
import { dashboardAPI } from "@/lib/api"

interface InstallmentData {
    month: string
    amount: number
}

export function MonthlyInstallmentsChart() {
    const [data, setData] = useState<InstallmentData[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        async function loadData() {
            try {
                const response = await dashboardAPI.getMonthlyInstallments()
                setData(response as unknown as InstallmentData[])
            } catch (error) {
                console.error("Failed to load installment data:", error)
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
                    <CardTitle>Monthly Installments Collected</CardTitle>
                </CardHeader>
                <CardContent className="h-[300px] flex items-center justify-center">
                    <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
                </CardContent>
            </Card>
        )
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.35 }}
            className="col-span-4"
        >
            <Card className="h-full">
                <CardHeader>
                    <CardTitle>Monthly Installments Collected</CardTitle>
                </CardHeader>
                <CardContent className="pl-2">
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={data}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                                <XAxis
                                    dataKey="month"
                                    stroke="#888888"
                                    fontSize={12}
                                    tickLine={false}
                                    axisLine={false}
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
                                    cursor={{ fill: 'rgba(79, 70, 229, 0.1)' }}
                                    formatter={(value: number) => [`€${value}`, "Amount"]}
                                />
                                <Bar
                                    dataKey="amount"
                                    fill="#4F46E5"
                                    radius={[8, 8, 0, 0]}
                                />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </CardContent>
            </Card>
        </motion.div>
    )
}
