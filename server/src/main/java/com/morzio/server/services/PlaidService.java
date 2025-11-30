package com.morzio.server.services;

import com.plaid.client.request.PlaidApi;
import com.plaid.client.model.*;
import org.springframework.stereotype.Service;
import retrofit2.Response;

import java.io.IOException;
import java.time.LocalDate;

import java.util.Collections;

@Service
public class PlaidService {

    private final PlaidApi plaidApi;

    public PlaidService(PlaidApi plaidApi) {
        this.plaidApi = plaidApi;
    }

    public String createLinkToken(String clientUserId) throws IOException {
        LinkTokenCreateRequestUser user = new LinkTokenCreateRequestUser()
                .clientUserId(clientUserId);

        LinkTokenCreateRequest request = new LinkTokenCreateRequest()
                .user(user)
                .clientName("Morzio")
                .products(Collections.singletonList(Products.TRANSACTIONS))
                .countryCodes(Collections.singletonList(CountryCode.GB))
                .language("en");

        Response<LinkTokenCreateResponse> response = plaidApi.linkTokenCreate(request).execute();

        if (!response.isSuccessful()) {
            throw new RuntimeException("Failed to create link token: "
                    + (response.errorBody() != null ? response.errorBody().string() : "Unknown error"));
        }

        return response.body().getLinkToken();
    }

    public String exchangePublicToken(String publicToken) throws IOException {
        ItemPublicTokenExchangeRequest request = new ItemPublicTokenExchangeRequest()
                .publicToken(publicToken);

        Response<ItemPublicTokenExchangeResponse> response = plaidApi.itemPublicTokenExchange(request).execute();

        if (!response.isSuccessful()) {
            throw new RuntimeException("Failed to exchange public token: "
                    + (response.errorBody() != null ? response.errorBody().string() : "Unknown error"));
        }

        return response.body().getAccessToken();
    }

    public TransactionsGetResponse getTransactions(String accessToken) throws IOException {
        LocalDate startDate = LocalDate.now().minusDays(90);
        LocalDate endDate = LocalDate.now();

        TransactionsGetRequest request = new TransactionsGetRequest()
                .accessToken(accessToken)
                .startDate(startDate)
                .endDate(endDate);

        Response<TransactionsGetResponse> response = plaidApi.transactionsGet(request).execute();

        if (!response.isSuccessful()) {
            throw new RuntimeException("Failed to get transactions: "
                    + (response.errorBody() != null ? response.errorBody().string() : "Unknown error"));
        }

        return response.body();
    }

    public AccountsGetResponse getAccountBalances(String accessToken) throws IOException {
        AccountsGetRequest request = new AccountsGetRequest()
                .accessToken(accessToken);

        Response<AccountsGetResponse> response = plaidApi.accountsGet(request).execute();

        if (!response.isSuccessful()) {
            throw new RuntimeException("Failed to get account balances: "
                    + (response.errorBody() != null ? response.errorBody().string() : "Unknown error"));
        }

        return response.body();
    }
}
