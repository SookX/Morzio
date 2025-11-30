"use client"

import { BackendStatus } from "./BackendStatus"

export function ClientProviders({ children }: { children: React.ReactNode }) {
    return (
        <>
            <BackendStatus />
            {children}
        </>
    )
}
