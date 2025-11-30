"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { motion } from "framer-motion"
import { useEffect, useState } from "react"
import { dashboardAPI } from "@/lib/api"
import { formatCurrency } from "@/lib/mockData"

interface Transaction {
    id: string
    customerName: string
    amount: number
    status: string
    timestamp: string
}

export function RecentTransactions() {
    const [transactions, setTransactions] = useState<Transaction[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        async function loadData() {
            try {
                const response = await dashboardAPI.getRecentTransactions()
                setTransactions(response as unknown as Transaction[])
            } catch (error) {
                console.error("Failed to load transactions:", error)
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
                    <CardTitle>Recent Transactions</CardTitle>
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
            transition={{ duration: 0.5, delay: 0.4 }}
            className="col-span-4 lg:col-span-3"
        >
            <Card className="h-full">
                <CardHeader>
                    <CardTitle>Recent Transactions</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="space-y-6">
                        {transactions.map((tx) => (
                            <div key={tx.id} className="flex items-center justify-between">
                                <div className="flex items-center space-x-4">
                                    <div className="h-10 w-10 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-700 font-bold text-sm">
                                        {tx.customerName.split('-')[1]?.substring(0, 2) || 'TX'}
                                    </div>
                                    <div>
                                        <p className="text-sm font-medium leading-none">{tx.customerName}</p>
                                        <p className="text-xs text-muted-foreground mt-1">
                                            {new Date(tx.timestamp).toLocaleString()}
                                        </p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className="text-sm font-bold">{formatCurrency(tx.amount)}</p>
                                    <p className={`text-xs ${tx.status === "completed" ? "text-green-600" :
                                            tx.status === "pending" ? "text-amber-600" :
                                                "text-red-600"
                                        }`}>
                                        {tx.status}
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>
        </motion.div>
    )
}
