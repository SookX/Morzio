"use client"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { motion } from "framer-motion"

const data = [
    { month: 'Jan', newCustomers: 45, activeCustomers: 120 },
    { month: 'Feb', newCustomers: 52, activeCustomers: 160 },
    { month: 'Mar', newCustomers: 38, activeCustomers: 190 },
    { month: 'Apr', newCustomers: 65, activeCustomers: 240 },
    { month: 'May', newCustomers: 48, activeCustomers: 280 },
    { month: 'Jun', newCustomers: 59, activeCustomers: 330 },
]

export function CustomerGrowthChart() {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="col-span-4 lg:col-span-2"
        >
            <Card className="h-full">
                <CardHeader>
                    <CardTitle>Customer Growth</CardTitle>
                    <CardDescription>New vs Active Customers</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={data}>
                                <CartesianGrid stroke="#f5f5f5" vertical={false} />
                                <XAxis dataKey="month" scale="band" fontSize={12} tickLine={false} axisLine={false} />
                                <YAxis fontSize={12} tickLine={false} axisLine={false} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: "#fff", borderRadius: "8px", border: "1px solid #E5E7EB" }}
                                />
                                <Legend verticalAlign="top" height={36} />
                                <Bar dataKey="newCustomers" barSize={20} fill="#4F46E5" radius={[4, 4, 0, 0]} name="New" />
                                <Line type="monotone" dataKey="activeCustomers" stroke="#10B981" strokeWidth={3} name="Total Active" />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>
                </CardContent>
            </Card>
        </motion.div>
    )
}
