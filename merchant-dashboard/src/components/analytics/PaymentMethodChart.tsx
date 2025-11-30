"use client"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts"
import { motion } from "framer-motion"

const data = [
    { name: 'Credit Card', value: 400 },
    { name: 'Debit Card', value: 300 },
    { name: 'Bank Transfer', value: 300 },
    { name: 'Digital Wallet', value: 200 },
]

const COLORS = ['#4F46E5', '#10B981', '#F59E0B', '#6366F1']

export function PaymentMethodChart() {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="col-span-4 lg:col-span-2"
        >
            <Card className="h-full">
                <CardHeader>
                    <CardTitle>Payment Methods</CardTitle>
                    <CardDescription>Preferred payment options</CardDescription>
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
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {data.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{ backgroundColor: "#fff", borderRadius: "8px", border: "1px solid #E5E7EB" }}
                                />
                                <Legend verticalAlign="bottom" height={36} iconType="circle" />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </CardContent>
            </Card>
        </motion.div>
    )
}
