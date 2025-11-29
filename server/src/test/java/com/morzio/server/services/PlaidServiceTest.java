package com.morzio.server.services;

import com.plaid.client.model.*;
import com.plaid.client.request.PlaidApi;
import okhttp3.ResponseBody;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import retrofit2.Call;
import retrofit2.Response;

import java.io.IOException;
import java.time.LocalDate;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class PlaidServiceTest {

    @Mock
    private PlaidApi plaidApi;

    @InjectMocks
    private PlaidService plaidService;

    @Captor
    private ArgumentCaptor<LinkTokenCreateRequest> linkTokenCreateRequestCaptor;

    @Captor
    private ArgumentCaptor<ItemPublicTokenExchangeRequest> itemPublicTokenExchangeRequestCaptor;

    @Captor
    private ArgumentCaptor<TransactionsGetRequest> transactionsGetRequestCaptor;

    @Test
    void createLinkToken_Success() throws IOException {
        String clientUserId = "user123";
        String expectedLinkToken = "link-sandbox-123";

        LinkTokenCreateResponse linkTokenCreateResponse = new LinkTokenCreateResponse()
                .linkToken(expectedLinkToken);

        Call<LinkTokenCreateResponse> mockCall = mock(Call.class);
        Response<LinkTokenCreateResponse> response = Response.success(linkTokenCreateResponse);

        when(plaidApi.linkTokenCreate(any(LinkTokenCreateRequest.class))).thenReturn(mockCall);
        when(mockCall.execute()).thenReturn(response);

        String result = plaidService.createLinkToken(clientUserId);

        assertEquals(expectedLinkToken, result);
        verify(plaidApi).linkTokenCreate(linkTokenCreateRequestCaptor.capture());
        verify(mockCall).execute();

        LinkTokenCreateRequest capturedRequest = linkTokenCreateRequestCaptor.getValue();
        assertEquals(clientUserId, capturedRequest.getUser().getClientUserId());
        assertEquals("Morzio", capturedRequest.getClientName());
        assertEquals(Collections.singletonList(Products.TRANSACTIONS), capturedRequest.getProducts());
        assertEquals(Collections.singletonList(CountryCode.GB), capturedRequest.getCountryCodes());
        assertEquals("en", capturedRequest.getLanguage());
    }

    @Test
    void createLinkToken_Failure() throws IOException {
        String clientUserId = "user123";

        Call<LinkTokenCreateResponse> mockCall = mock(Call.class);
        ResponseBody errorBody = ResponseBody.create(null, "Bad Request");
        Response<LinkTokenCreateResponse> response = Response.error(400, errorBody);

        when(plaidApi.linkTokenCreate(any(LinkTokenCreateRequest.class))).thenReturn(mockCall);
        when(mockCall.execute()).thenReturn(response);

        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            plaidService.createLinkToken(clientUserId);
        });

        assertTrue(exception.getMessage().contains("Failed to create link token"));
    }

    @Test
    void exchangePublicToken_Success() throws IOException {
        String publicToken = "public-sandbox-123";
        String expectedAccessToken = "access-sandbox-123";

        ItemPublicTokenExchangeResponse exchangeResponse = new ItemPublicTokenExchangeResponse()
                .accessToken(expectedAccessToken);

        Call<ItemPublicTokenExchangeResponse> mockCall = mock(Call.class);
        Response<ItemPublicTokenExchangeResponse> response = Response.success(exchangeResponse);

        when(plaidApi.itemPublicTokenExchange(any(ItemPublicTokenExchangeRequest.class))).thenReturn(mockCall);
        when(mockCall.execute()).thenReturn(response);

        String result = plaidService.exchangePublicToken(publicToken);

        assertEquals(expectedAccessToken, result);
        verify(plaidApi).itemPublicTokenExchange(itemPublicTokenExchangeRequestCaptor.capture());

        ItemPublicTokenExchangeRequest capturedRequest = itemPublicTokenExchangeRequestCaptor.getValue();
        assertEquals(publicToken, capturedRequest.getPublicToken());
    }

    @Test
    void exchangePublicToken_Failure() throws IOException {
        String publicToken = "public-sandbox-123";

        Call<ItemPublicTokenExchangeResponse> mockCall = mock(Call.class);
        ResponseBody errorBody = ResponseBody.create(null, "Invalid Token");
        Response<ItemPublicTokenExchangeResponse> response = Response.error(400, errorBody);

        when(plaidApi.itemPublicTokenExchange(any(ItemPublicTokenExchangeRequest.class))).thenReturn(mockCall);
        when(mockCall.execute()).thenReturn(response);

        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            plaidService.exchangePublicToken(publicToken);
        });

        assertTrue(exception.getMessage().contains("Failed to exchange public token"));
    }

    @Test
    void getTransactions_Success() throws IOException {
        String accessToken = "access-sandbox-123";
        TransactionsGetResponse expectedResponse = new TransactionsGetResponse();

        Call<TransactionsGetResponse> mockCall = mock(Call.class);
        Response<TransactionsGetResponse> response = Response.success(expectedResponse);

        when(plaidApi.transactionsGet(any(TransactionsGetRequest.class))).thenReturn(mockCall);
        when(mockCall.execute()).thenReturn(response);

        TransactionsGetResponse result = plaidService.getTransactions(accessToken);

        assertNotNull(result);
        assertEquals(expectedResponse, result);
        verify(plaidApi).transactionsGet(transactionsGetRequestCaptor.capture());

        TransactionsGetRequest capturedRequest = transactionsGetRequestCaptor.getValue();
        assertEquals(accessToken, capturedRequest.getAccessToken());
        assertEquals(LocalDate.now(), capturedRequest.getEndDate());
        assertEquals(LocalDate.now().minusDays(90), capturedRequest.getStartDate());
    }

    @Test
    void getTransactions_Failure() throws IOException {
        String accessToken = "access-sandbox-123";

        Call<TransactionsGetResponse> mockCall = mock(Call.class);
        ResponseBody errorBody = ResponseBody.create(null, "Server Error");
        Response<TransactionsGetResponse> response = Response.error(500, errorBody);

        when(plaidApi.transactionsGet(any(TransactionsGetRequest.class))).thenReturn(mockCall);
        when(mockCall.execute()).thenReturn(response);

        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            plaidService.getTransactions(accessToken);
        });

        assertTrue(exception.getMessage().contains("Failed to get transactions"));
    }
}
