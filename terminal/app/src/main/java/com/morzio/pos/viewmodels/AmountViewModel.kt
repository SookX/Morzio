package com.morzio.pos.viewmodels

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.util.Locale

data class AmountState(
    val amount: String = "0.00",
    val isValid: Boolean = false
)

class AmountViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(AmountState())
    val uiState: StateFlow<AmountState> = _uiState.asStateFlow()

    private var amountInCents = 0L

    fun onKeyPress(key: String) {
        when (key) {
            "←" -> {
                amountInCents /= 10
            }
            else -> {
                val digit = key.toLongOrNull()
                if (digit != null) {
                    val newAmount = amountInCents * 10 + digit
                    if (newAmount <= 999999) { // Limit to €9,999.99
                        amountInCents = newAmount
                    }
                }
            }
        }

        val formattedAmount = formatAmount(amountInCents)
        val amountAsDouble = amountInCents / 100.0
        val isValid = amountAsDouble >= 0.01

        _uiState.update {
            it.copy(
                amount = formattedAmount,
                isValid = isValid
            )
        }
    }

    private fun formatAmount(cents: Long): String {
        val amount = cents / 100.0
        return String.format(Locale.GERMANY, "%.2f", amount)
    }
}