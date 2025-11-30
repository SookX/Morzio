"use client"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
    LayoutDashboard,
    CreditCard,
    PieChart,
    LogOut,
    Menu
} from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"

const sidebarItems = [
    { icon: LayoutDashboard, label: "Dashboard", href: "/" },
    { icon: CreditCard, label: "Payments", href: "/payments" },
    { icon: PieChart, label: "Analytics", href: "/analytics" },
]

export function Sidebar() {
    const pathname = usePathname()
    const [isMobileOpen, setIsMobileOpen] = useState(false)
    const [isDesktop, setIsDesktop] = useState(false)

    // Handle responsive behavior to ensure sidebar behaves correctly on resize
    useEffect(() => {
        const checkDesktop = () => {
            setIsDesktop(window.innerWidth >= 768)
        }

        checkDesktop()
        window.addEventListener('resize', checkDesktop)
        return () => window.removeEventListener('resize', checkDesktop)
    }, [])

    return (
        <>
            <div className="md:hidden fixed top-4 left-4 z-50">
                <Button variant="outline" size="icon" onClick={() => setIsMobileOpen(!isMobileOpen)}>
                    <Menu className="h-5 w-5" />
                </Button>
            </div>

            <AnimatePresence>
                {(isMobileOpen || isDesktop) && (
                    <motion.aside
                        initial={{ x: -300 }}
                        animate={{ x: 0 }}
                        exit={{ x: -300 }}
                        className={cn(
                            "fixed inset-y-0 left-0 z-40 w-64 bg-card border-r border-border shadow-sm transform transition-transform duration-300 ease-in-out md:translate-x-0 md:static md:h-screen",
                            isMobileOpen ? "translate-x-0" : "-translate-x-full"
                        )}
                    >
                        <div className="flex flex-col h-full">
                            <div className="h-20 flex items-center justify-center border-b border-border/50">
                                <div className="flex items-center gap-2.5 opacity-90">
                                    <div className="w-8 h-8 bg-black rounded-xl flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-black/10">
                                        M
                                    </div>
                                    <span className="font-bold text-xl tracking-tight text-gray-900">Morzio</span>
                                </div>
                            </div>

                            <nav className="flex-1 px-4 py-6 space-y-2">
                                {sidebarItems.map((item) => {
                                    const isActive = pathname === item.href
                                    return (
                                        <Link key={item.href} href={item.href} onClick={() => setIsMobileOpen(false)}>
                                            <div
                                                className={cn(
                                                    "flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200",
                                                    isActive
                                                        ? "bg-primary text-primary-foreground shadow-md"
                                                        : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                                                )}
                                            >
                                                <item.icon className={cn("h-5 w-5", isActive ? "text-primary-foreground" : "text-muted-foreground")} />
                                                {item.label}
                                            </div>
                                        </Link>
                                    )
                                })}
                            </nav>

                            <div className="p-4 border-t border-border">
                                <Button variant="ghost" className="w-full justify-start gap-3 text-red-500 hover:text-red-600 hover:bg-red-50">
                                    <LogOut className="h-5 w-5" />
                                    Logout
                                </Button>
                            </div>
                        </div>
                    </motion.aside>
                )}
            </AnimatePresence>

            {isMobileOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-30 md:hidden"
                    onClick={() => setIsMobileOpen(false)}
                />
            )}
        </>
    )
}
