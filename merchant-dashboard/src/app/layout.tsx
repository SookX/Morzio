import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { ClientProviders } from "@/components/ClientProviders";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Morzio Merchant Dashboard",
  description: "Manage your payments and installments with ease.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-background text-foreground antialiased`}>
        <div className="flex h-screen overflow-hidden">
          <Sidebar />
          <div className="flex-1 flex flex-col h-full overflow-hidden">
            <Header />
            <main className="flex-1 overflow-y-auto p-6 md:p-8">
              <ClientProviders>{children}</ClientProviders>
            </main>
          </div>
        </div>
      </body>
    </html>
  );
}
