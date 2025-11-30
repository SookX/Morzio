"use client"

import { PaymentsStats } from "@/components/payments/PaymentsStats"
import { PaymentsTable } from "@/components/payments/PaymentsTable"

export default function PaymentsPage() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold tracking-tight">Payments</h1>
                <p className="text-muted-foreground mt-1">
                    Manage all payment transactions and links.
                </p>
            </div>

            {/* Statistics */}
            <PaymentsStats />

            {/* Payments Table */}
            <PaymentsTable />
        </div>
    )
}
