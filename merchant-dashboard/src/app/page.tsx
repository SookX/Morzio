"use client"

import { MetricsCards } from "@/components/dashboard/MetricsCards"
import { RevenueChart } from "@/components/dashboard/RevenueChart"
import { PaymentPlanCard } from "@/components/dashboard/PaymentPlanCard"
import { RecentTransactions } from "@/components/dashboard/RecentTransactions"
import { PaymentStatusChart } from "@/components/dashboard/PaymentStatusChart"
import { MonthlyInstallmentsChart } from "@/components/dashboard/MonthlyInstallmentsChart"
import { useEffect, useState } from "react"
import { dashboardAPI } from "@/lib/api"

interface PaymentPlan {
  id: string
  customerName: string
  totalAmount: number
  installmentCount: number
  paidCount: number
  nextDueDate: string
  status: string
}

export default function Home() {
  const [plans, setPlans] = useState<PaymentPlan[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function loadPlans() {
      try {
        const response = await dashboardAPI.getPaymentPlans("Active", 3)
        setPlans(response as unknown as PaymentPlan[])
      } catch (error) {
        console.error("Failed to load payment plans:", error)
      } finally {
        setLoading(false)
      }
    }
    loadPlans()
  }, [])

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground mt-1">
          Welcome back, here&apos;s what&apos;s happening with your payments today.
        </p>
      </div>

      {/* Metrics */}
      <MetricsCards />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-7 gap-8">
        <RevenueChart />
        <RecentTransactions />
      </div>

      {/* Secondary Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-7 gap-8">
        <MonthlyInstallmentsChart />
        <PaymentStatusChart />
      </div>

      {/* Active Plans Section */}
      <div className="space-y-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight">Active Payment Plans</h2>
        </div>

        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-48 bg-gray-100 animate-pulse rounded-xl" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {plans.map((plan, index) => (
              <PaymentPlanCard
                key={plan.id}
                id={plan.id}
                customerName={plan.customerName}
                totalAmount={plan.totalAmount}
                installmentCount={plan.installmentCount}
                paidCount={plan.paidCount}
                nextDueDate={plan.nextDueDate}
                status={plan.status}
                index={index}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
