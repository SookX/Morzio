"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertCircle, CheckCircle, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"

export function BackendStatus() {
    const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking')
    const [lastChecked, setLastChecked] = useState<Date | null>(null)

    const checkBackendHealth = async () => {
        setStatus('checking')
        try {
            const response = await fetch('http://localhost:8080/api/dashboard/metrics', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            })

            if (response.ok) {
                setStatus('online')
            } else {
                setStatus('offline')
            }
        } catch (error) {
            console.error('Backend health check failed:', error)
            setStatus('offline')
        }
        setLastChecked(new Date())
    }

    useEffect(() => {
        checkBackendHealth()
        // Check every 30 seconds
        const interval = setInterval(checkBackendHealth, 30000)
        return () => clearInterval(interval)
    }, [])

    if (status === 'online') return null

    return (
        <Card className="mb-4 border-l-4 border-l-amber-500">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-amber-700">
                    {status === 'checking' ? (
                        <>
                            <RefreshCw className="h-5 w-5 animate-spin" />
                            Checking backend connection...
                        </>
                    ) : (
                        <>
                            <AlertCircle className="h-5 w-5" />
                            Backend Connection Issue
                        </>
                    )}
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="space-y-3">
                    <p className="text-sm text-muted-foreground">
                        {status === 'offline' && (
                            <>
                                Unable to connect to the backend server at <code className="px-1 py-0.5 bg-muted rounded">http://localhost:8080</code>
                            </>
                        )}
                    </p>
                    {status === 'offline' && (
                        <div className="space-y-2">
                            <p className="text-sm font-medium">To fix this:</p>
                            <ol className="list-decimal list-inside text-sm text-muted-foreground space-y-1">
                                <li>Ensure the Spring Boot backend is running on port 8080</li>
                                <li>Check that CORS is configured to allow <code className="px-1 py-0.5 bg-muted rounded">http://localhost:3000</code></li>
                                <li>Verify the database connection is working</li>
                            </ol>
                            <Button
                                onClick={checkBackendHealth}
                                variant="outline"
                                size="sm"
                                className="mt-3"
                            >
                                <RefreshCw className="h-4 w-4 mr-2" />
                                Retry Connection
                            </Button>
                        </div>
                    )}
                    {lastChecked && (
                        <p className="text-xs text-muted-foreground mt-2">
                            Last checked: {lastChecked.toLocaleTimeString()}
                        </p>
                    )}
                </div>
            </CardContent>
        </Card>
    )
}
