"use client"

import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { motion } from "framer-motion"
import { ArrowRight, Calendar } from "lucide-react"
import { formatCurrency } from "@/lib/mockData"

interface PaymentPlanProps {
    id: string
    customerName: string
    totalAmount: number
    installmentCount: number
    paidCount: number
    nextDueDate: string
    status: string
    index: number
}

export function PaymentPlanCard({
    customerName,
    totalAmount,
    installmentCount,
    paidCount,
    nextDueDate,
    status,
    index
}: PaymentPlanProps) {
    const progress = (paidCount / installmentCount) * 100

    return (
        <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.3 + index * 0.1 }}
        >
            <Card className="overflow-hidden hover:shadow-lg transition-all duration-300 border-l-4 border-l-indigo-500">
                <CardHeader className="pb-2">
                    <div className="flex justify-between items-start">
                        <div>
                            <CardTitle className="text-lg">{customerName}</CardTitle>
                            <p className="text-sm text-muted-foreground mt-1">Plan ID: #{index + 1000}</p>
                        </div>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${status === "Active" ? "bg-blue-100 text-blue-700" :
                                status === "Completed" ? "bg-green-100 text-green-700" :
                                    "bg-red-100 text-red-700"
                            }`}>
                            {status}
                        </span>
                    </div>
                </CardHeader>
                <CardContent className="pb-2">
                    <div className="mt-2 mb-4">
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-muted-foreground">Progress</span>
                            <span className="font-medium">{paidCount}/{installmentCount} Paid</span>
                        </div>
                        <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                            <div
                                className="h-full bg-indigo-500 rounded-full transition-all duration-1000 ease-out"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                            <p className="text-muted-foreground text-xs">Total Amount</p>
                            <p className="font-bold text-lg">{formatCurrency(totalAmount)}</p>
                        </div>
                        <div>
                            <p className="text-muted-foreground text-xs">Next Due</p>
                            <div className="flex items-center gap-1 font-medium">
                                <Calendar className="h-3 w-3 text-indigo-500" />
                                {nextDueDate}
                            </div>
                        </div>
                    </div>
                </CardContent>
                <CardFooter className="pt-2">
                    <Button variant="ghost" className="w-full justify-between group text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50">
                        View Details
                        <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
                    </Button>
                </CardFooter>
            </Card>
        </motion.div>
    )
}
