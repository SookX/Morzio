"use client"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from "recharts"
import { motion } from "framer-motion"
import { useState, useEffect } from "react"

// Mock data for detailed analysis - in a real app this would come from an endpoint like /api/analytics/revenue
const data = [
    { date: "Nov 01", revenue: 4000, previous: 2400 },
    { date: "Nov 05", revenue: 3000, previous: 1398 },
    { date: "Nov 10", revenue: 2000, previous: 9800 },
    { date: "Nov 15", revenue: 2780, previous: 3908 },
    { date: "Nov 20", revenue: 1890, previous: 4800 },
    { date: "Nov 25", revenue: 2390, previous: 3800 },
    { date: "Nov 30", revenue: 3490, previous: 4300 },
]

export function DetailedRevenueChart() {
    const [mounted, setMounted] = useState(false)

    useEffect(() => {
        setMounted(true)
    }, [])

    if (!mounted) return null

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="col-span-4"
        >
            <Card className="h-full">
                <CardHeader>
                    <CardTitle>Revenue Trends</CardTitle>
                    <CardDescription>Comparison with previous period</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="h-[400px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="colorRevenue" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#4F46E5" stopOpacity={0.8} />
                                        <stop offset="95%" stopColor="#4F46E5" stopOpacity={0} />
                                    </linearGradient>
                                    <linearGradient id="colorPrevious" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#94A3B8" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#94A3B8" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <XAxis dataKey="date" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                                <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `€${value}`} />
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                                <Tooltip
                                    contentStyle={{ backgroundColor: "#fff", borderRadius: "8px", border: "1px solid #E5E7EB", boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)" }}
                                />
                                <Area type="monotone" dataKey="revenue" stroke="#4F46E5" fillOpacity={1} fill="url(#colorRevenue)" strokeWidth={3} />
                                <Area type="monotone" dataKey="previous" stroke="#94A3B8" fillOpacity={1} fill="url(#colorPrevious)" strokeWidth={2} strokeDasharray="5 5" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </CardContent>
            </Card>
        </motion.div>
    )
}
