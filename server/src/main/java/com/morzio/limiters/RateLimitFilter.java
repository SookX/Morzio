package com.morzio.limiters;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * limiter for requests for secutiry reasons 10 requests per minute
 */

@Component
public class RateLimitFilter extends OncePerRequestFilter {

    private final Map<String, RequestInfo> requestCounts = new ConcurrentHashMap<>();
    private final int MAX_REQUESTS = 10;
    private final long TIME_WINDOW = 60000;

    private static class RequestInfo {
        AtomicInteger count = new AtomicInteger(0);
        long firstRequestTime = System.currentTimeMillis();
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain filterChain)
            throws ServletException, IOException {

        String ip = request.getRemoteAddr();
        RequestInfo info = requestCounts.computeIfAbsent(ip, k -> new RequestInfo());

        synchronized (info) {
            long now = System.currentTimeMillis();

            if (now - info.firstRequestTime > TIME_WINDOW) {
                info.count.set(0);
                info.firstRequestTime = now;
            }

            if (info.count.incrementAndGet() > MAX_REQUESTS) {
                response.setStatus(429);
                response.getWriter().write("Too many requests");
                return;
            }
        }

        filterChain.doFilter(request, response);
    }
}
