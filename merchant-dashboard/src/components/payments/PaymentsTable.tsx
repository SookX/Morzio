"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { motion } from "framer-motion"
import {
    Search,
    Filter,
    Download,
    CheckCircle,
    Clock,
    XCircle,
    ArrowUpDown,
    Eye
} from "lucide-react"
import { formatCurrency } from "@/lib/mockData"
import { dashboardAPI } from "@/lib/api"

interface Payment {
    id: string
    customerName: string
    amount: number
    status: string
    timestamp: string
    method?: string
}

const statusConfig = {
    completed: { label: "Completed", color: "bg-green-100 text-green-700", icon: CheckCircle },
    pending: { label: "Pending", color: "bg-amber-100 text-amber-700", icon: Clock },
    failed: { label: "Failed", color: "bg-red-100 text-red-700", icon: XCircle },
}

export function PaymentsTable() {
    const [payments, setPayments] = useState<Payment[]>([])
    const [allPayments, setAllPayments] = useState<Payment[]>([])
    const [loading, setLoading] = useState(true)
    const [filter, setFilter] = useState<string>("all")
    const [searchQuery, setSearchQuery] = useState("")

    // Grab all payments first so we can calculate those nice little badge counts
    useEffect(() => {
        async function loadAllPayments() {
            try {
                const response = await dashboardAPI.getRecentTransactions(1000)
                setAllPayments(response as unknown as Payment[])
            } catch (error) {
                console.error("Failed to load all payments:", error)
            }
        }
        loadAllPayments()
    }, [])

    // When the user switches tabs, we fetch fresh data from the backend
    useEffect(() => {
        async function loadPayments() {
            setLoading(true)
            try {
                const response = await dashboardAPI.getRecentTransactions(100, filter === "all" ? undefined : filter)
                setPayments(response as unknown as Payment[])
            } catch (error) {
                console.error("Failed to load payments:", error)
            } finally {
                setLoading(false)
            }
        }
        loadPayments()
    }, [filter])

    const filteredPayments = payments.filter(payment => {
        const matchesSearch = payment.customerName.toLowerCase().includes(searchQuery.toLowerCase()) ||
            payment.id.toLowerCase().includes(searchQuery.toLowerCase())
        return matchesSearch
    })

    const stats = {
        all: allPayments.length,
        completed: allPayments.filter(p => p.status.toLowerCase() === "completed").length,
        pending: allPayments.filter(p => p.status.toLowerCase() === "pending").length,
        failed: allPayments.filter(p => p.status.toLowerCase() === "failed").length,
    }

    return (
        <div className="space-y-6">
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
                <div className="relative w-full sm:w-96">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <input
                        type="text"
                        placeholder="Search by customer or ID..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full h-10 pl-10 pr-4 rounded-lg bg-secondary border-none text-sm focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                    />
                </div>
                <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                        <Filter className="mr-2 h-4 w-4" />
                        Filters
                    </Button>
                    <Button variant="outline" size="sm">
                        <Download className="mr-2 h-4 w-4" />
                        Export
                    </Button>
                </div>
            </div>

            <div className="flex gap-2 overflow-x-auto pb-2">
                {[
                    { key: "all", label: "All Payments" },
                    { key: "completed", label: "Completed" },
                    { key: "pending", label: "Pending" },
                    { key: "failed", label: "Failed" },
                ].map((tab) => (
                    <Button
                        key={tab.key}
                        variant={filter === tab.key ? "default" : "outline"}
                        size="sm"
                        onClick={() => setFilter(tab.key)}
                        className="whitespace-nowrap"
                    >
                        {tab.label}
                        <span className="ml-2 px-2 py-0.5 rounded-full bg-white/20 text-xs">
                            {stats[tab.key as keyof typeof stats]}
                        </span>
                    </Button>
                ))}
            </div>

            <Card>
                <CardContent className="p-0">
                    {loading ? (
                        <div className="flex items-center justify-center h-64">
                            <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
                        </div>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead className="bg-muted/50">
                                    <tr className="border-b">
                                        <th className="text-left p-4 font-medium text-sm">
                                            <div className="flex items-center gap-2">
                                                Transaction ID
                                                <ArrowUpDown className="h-3 w-3 text-muted-foreground cursor-pointer" />
                                            </div>
                                        </th>
                                        <th className="text-left p-4 font-medium text-sm">Customer</th>
                                        <th className="text-left p-4 font-medium text-sm">Amount</th>
                                        <th className="text-left p-4 font-medium text-sm">Status</th>
                                        <th className="text-left p-4 font-medium text-sm">Date</th>
                                        <th className="text-left p-4 font-medium text-sm">Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredPayments.map((payment, index) => {
                                        const statusKey = payment.status.toLowerCase() as keyof typeof statusConfig
                                        const StatusIcon = statusConfig[statusKey]?.icon || Clock
                                        const statusStyle = statusConfig[statusKey]?.color || "bg-gray-100 text-gray-700"

                                        return (
                                            <motion.tr
                                                key={payment.id}
                                                initial={{ opacity: 0, y: 10 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                transition={{ delay: index * 0.02 }}
                                                className="border-b hover:bg-muted/30 transition-colors"
                                            >
                                                <td className="p-4 text-sm font-mono">{payment.id}</td>
                                                <td className="p-4 text-sm">{payment.customerName}</td>
                                                <td className="p-4 text-sm font-semibold">{formatCurrency(payment.amount)}</td>
                                                <td className="p-4">
                                                    <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium ${statusStyle}`}>
                                                        <StatusIcon className="h-3 w-3" />
                                                        {payment.status}
                                                    </span>
                                                </td>
                                                <td className="p-4 text-sm text-muted-foreground">
                                                    {new Date(payment.timestamp).toLocaleDateString()}
                                                </td>
                                                <td className="p-4">
                                                    <Button variant="ghost" size="sm">
                                                        <Eye className="h-4 w-4" />
                                                    </Button>
                                                </td>
                                            </motion.tr>
                                        )
                                    })}
                                </tbody>
                            </table>
                        </div>
                    )}
                </CardContent>
            </Card>

            <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                    Showing {filteredPayments.length} {filter !== "all" ? `${filter} ` : ""}payments
                    {searchQuery && ` matching "${searchQuery}"`}
                </p>
                <div className="flex gap-2">
                    <Button variant="outline" size="sm" disabled>Previous</Button>
                    <Button variant="outline" size="sm" disabled>Next</Button>
                </div>
            </div>
        </div>
    )
}
