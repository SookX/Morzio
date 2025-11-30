"use client"

import { Button } from "@/components/ui/button"
import { Bell, Search } from "lucide-react"

export function Header() {
    return (
        <header className="h-16 border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-30 px-6 flex items-center justify-between pl-14 md:pl-6">
            <div className="flex items-center gap-4 w-full max-w-md hidden sm:flex">
                <div className="relative w-full">
                    <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                    <input
                        type="text"
                        placeholder="Search transactions, customers..."
                        className="w-full h-9 pl-9 pr-4 rounded-full bg-secondary border-none text-sm focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                    />
                </div>
            </div>
            <div className="sm:hidden flex items-center gap-2">
                <div className="w-7 h-7 bg-black rounded-lg flex items-center justify-center text-white font-bold text-sm shadow-md">
                    M
                </div>
                <span className="font-bold text-lg tracking-tight text-gray-900">Morzio</span>
            </div>

            <div className="flex items-center gap-4">
                <Button variant="ghost" size="icon" className="relative">
                    <Bell className="h-5 w-5 text-muted-foreground" />
                    <span className="absolute top-2 right-2 h-2 w-2 bg-red-500 rounded-full border-2 border-card" />
                </Button>
                <div className="h-8 w-8 rounded-full bg-indigo-100 border border-indigo-200 flex items-center justify-center text-indigo-700 font-bold text-xs cursor-pointer">
                    JD
                </div>
            </div>
        </header>
    )
}
