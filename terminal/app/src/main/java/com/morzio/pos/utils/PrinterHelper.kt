package com.morzio.pos.utils

import android.util.Log
import recieptservice.com.recieptservice.PrinterInterface
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object PrinterHelper {
    private const val TAG = "PrinterHelper"

    fun printReceipt(
        printerService: PrinterInterface,
        amount: Double,
        sessionId: String,
        status: String,
        paymentUrl: String,
        installments: List<com.morzio.pos.data.api.InstallmentDto>?
    ) {
        try {
            printerService.apply {
                setAlignment(1) // Center
                setTextSize(30f)
                setTextBold(true)
                printText("MORZIO")
                nextLine(1)
                setTextSize(24f)
                setTextBold(false)
                printText("Payment Receipt")
                nextLine(2)

                setAlignment(0) // Left
                setTextSize(24f)
                
                val amountString = String.format(Locale.US, "%.2f", amount)
                val currency = "EUR" 
                
                printTableText(
                    arrayOf("Amount", "$amountString $currency"),
                    intArrayOf(1, 1),
                    intArrayOf(0, 2) // Left, Right
                )
                
                printText("--------------------------------")
                nextLine(1)

                val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
                val dateStr = dateFormat.format(Date())

                val detailsText = arrayOf(
                    "Session ID:", sessionId.take(8) + "...",
                    "Status:", status,
                    "Date:", dateStr
                )
                
                setTextSize(20f)
                for (i in detailsText.indices step 2) {
                     printTableText(
                        arrayOf(detailsText[i], detailsText[i+1]),
                        intArrayOf(1, 1),
                        intArrayOf(0, 2)
                    )
                }

                if (!installments.isNullOrEmpty()) {
                    nextLine(1)
                    setTextBold(true)
                    printText("Installment Schedule:")
                    setTextBold(false)
                    nextLine(1)
                    
                    installments.forEachIndexed { index, installment ->
                        val amountStr = String.format(Locale.US, "%.2f", installment.amount / 100.0)
                        printTableText(
                            arrayOf("${index + 1}. ${installment.dueDate}", "$amountStr EUR"),
                            intArrayOf(2, 1),
                            intArrayOf(0, 2)
                        )
                    }
                }

                nextLine(1)
                printText("--------------------------------")
                nextLine(1)
                
                setAlignment(1)
                printQRCode("https://morzio.com", 5, 1)
                nextLine(3)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error printing receipt", e)
        }
    }
}
