# Morzio Merchant Dashboard - Comprehensive Overview

This is a fully responsive, modern, and visually impressive Next.js merchant dashboard for managing payments and installments.

## ✨ Features Implemented

### 1. **Fully Responsive Design**
- **Mobile-First Approach**: All components adapt seamlessly from mobile (320px) to desktop (1920px+)
- **Responsive Sidebar**: Collapsible sidebar with smooth animations for mobile devices
- **Adaptive Header**: Search bar hidden on mobile, logo displayed instead
- **Responsive Grids**: Charts and content automatically adjust to screen size
- **Touch-Optimized**: All interactive elements are sized appropriately for touch devices

### 2. **Dashboard Page** (`/`)
Built-in components with real-time data from backend:
- **Metrics Cards**: Total Revenue, Pending Installments, Active Plans, Completion Rate
- **Revenue Overview**: Line chart showing revenue trends
- **Recent Transactions**: Latest payment activities
- **Monthly Installments**: Bar chart for installment collection
- **Payment Status Distribution**: Pie chart visualization
- **Active Payment Plans**: Cards showing ongoing payment plans with progress bars

### 3. **Analytics Page** (`/analytics`)
Advanced data visualization with:
- **Analytics Metrics**: 4 key performance indicators with trend arrows
- **Revenue Trends**: Area chart with gradient fills comparing current vs previous period
- **Customer Growth**: Composed chart (Bar + Line) showing new vs active customers
- **Payment Methods**: Donut chart showing preferred payment options
- **Interactive Controls**: Date range selector and export functionality

### 4. **Payments Page** (`/payments`)
Comprehensive payment management:
- **Payment Statistics**: Volume, success rate, average transaction, total count
- **Smart Search**: Real-time filtering by customer name or transaction ID
- **Status Tabs**: Filter by All, Completed, Pending, Failed
- **Interactive Table**: Sortable columns with hover effects
- **Row Animations**: Staggered fade-in for smooth visual experience
- **Action Buttons**: View details, export, and filter options

## 🎨 Design System

### Color Palette
- **Primary**: Indigo (#4F46E5) - Main brand color
- **Success**: Green (#10B981) - Positive trends, completed states
- **Warning**: Amber (#F59E0B) - Pending states
- **Error**: Red (#EF4444) - Failed states, alerts
- **Backgrounds**: Gray scale with subtle variations

### Typography
- **Font**: Inter (Google Font) - Clean, modern, readable
- **Hierarchy**: Clear sizing from 3xl headers to xs captions
- **Weights**: 400 (regular), 500 (medium), 600 (semibold), 700 (bold)

### Visual Effects
- **Glassmorphism**: Header with `backdrop-blur-sm` and semi-transparent background
- **Micro-animations**: Hover effects on cards, buttons, and interactive elements
- **Gradients**: Used in charts (Area chart, payment status)
- **Shadows**: Subtle elevation for depth (cards hover: `shadow-lg`)
- **Smooth Transitions**: 150ms cubic-bezier easing on all interactive elements

## 🚀 Performance Optimizations

### Code Splitting
- Client-side components marked with `"use client"`
- Server components for static content
- Dynamic imports for heavy chart libraries

### Animation Performance
- Hardware-accelerated transforms (`translateX`, `translateY`, `scale`)
- `Framer Motion` for optimized React animations
- Respects `prefers-reduced-motion` for accessibility

### Data Loading
- Async data fetching with loading states
- Error boundaries for graceful failures
- Skeleton loaders for better perceived performance

### Bundle Optimization
- Tree-shaking for unused code
- Next.js automatic code splitting
- On-demand loading of routes

## 📊 Charts & Visualizations

### Recharts Library
All charts use the `recharts` library for consistency:

#### Line Chart (Dashboard - Revenue Overview)
```tsx
<LineChart>
  - Monotone curve for smooth lines
  - Custom tooltips with styled containers
  - Dot indicators on data points
  - Grid for easy reading
</LineChart>
```

#### Area Chart (Analytics - Revenue Trends)
```tsx
<AreaChart>
  - Linear gradients for fill
  - Comparison with previous period
  - Dashed line for historical data
  - Responsive container
</AreaChart>
```

#### Bar Chart (Dashboard - Monthly Installments)
```tsx
<BarChart>
  - Rounded corners on bars
  - Hover effects
  - Custom colors matching brand
  - Y-axis with currency formatting
</BarChart>
```

#### Pie Chart (Dashboard - Payment Status)
```tsx
<PieChart>
  - Inner radius for donut effect
  - Custom colors per status
  - Percentage labels
  - Interactive legend
</PieChart>
```

#### Composed Chart (Analytics - Customer Growth)
```tsx
<ComposedChart>
  - Bar + Line combination
  - Dual metrics visualization
  - Shared X-axis
  - Distinct colors for clarity
</ComposedChart>
```

## 🔧 Technical Stack

### Frontend
- **Framework**: Next.js 16 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **Animations**: Framer Motion
- **Charts**: Recharts
- **Icons**: Lucide React
- **Utilities**: clsx, tailwind-merge

### Backend Integration
- **API Client**: Custom fetch wrapper (`src/lib/api.ts`)
- **Endpoints**: RESTful API at `http://localhost:8080/api/`
- **Data Format**: JSON
- **Error Handling**: Try-catch with user-friendly messages

### Backend (Spring Boot)
- **Framework**: Spring Boot
- **Controllers**: DashboardController with 6 endpoints
- **Services**: DashboardService for business logic
- **DTOs**: Type-safe data transfer objects
- **Database**: PostgreSQL (via JPA/Hibernate)

## 📱 Responsive Breakpoints

```css
- Mobile: < 640px (sm)
- Tablet: 640px - 768px (md)
- Desktop: 768px - 1024px (lg)
- Large Desktop: > 1024px (xl)
```

### Layout Adaptations

#### Mobile (< 768px)
- Sidebar: Fixed overlay with toggle button
- Header: Logo only, no search bar
- Grids: Single column (grid-cols-1)
- Padding: Reduced (p-6)
- Tables: Horizontal scroll

#### Tablet (768px - 1024px)
- Sidebar: Static, always visible
- Header: Search bar visible
- Grids: 2 columns (grid-cols-2)
- Charts: Responsive with maintained aspect ratio

#### Desktop (> 1024px)
- Sidebar: Static with full navigation
- Header: Full search and controls
- Grids: 4 columns (grid-cols-4)
- All features visible

## 🎯 Key Components

### Layout Components
- `Sidebar.tsx`: Responsive navigation with mobile menu
- `Header.tsx`: Top navigation with search and user profile
- `layout.tsx`: Root layout wrapper

### Dashboard Components
- `MetricsCards.tsx`: KPI cards with trends
- `RevenueChart.tsx`: Revenue line chart
- `PaymentPlanCard.tsx`: Individual plan cards
- `RecentTransactions.tsx`: Transaction list
- `PaymentStatusChart.tsx`: Status pie chart
- `MonthlyInstallmentsChart.tsx`: Installment bar chart

### Analytics Components
- `AnalyticsMetrics.tsx`: Performance indicators
- `DetailedRevenueChart.tsx`: Advanced revenue visualization
- `CustomerGrowthChart.tsx`: Growth metrics
- `PaymentMethodChart.tsx`: Payment method distribution

### Payments Components
- `PaymentsStats.tsx`: Payment statistics
- `PaymentsTable.tsx`: Interactive payments table

## ⚡ Quick Start

### Development
```bash
# Frontend
cd merchant-dashboard
npm run dev
# Visit http://localhost:3000

# Backend
cd server
export $(grep -v '^#' .env | xargs)
export ALLOWED_ORIGINS="http://localhost:3000"
./mvnw spring-boot:run
# API at http://localhost:8080
```

### Production Build
```bash
npm run build
npm start
```

## 🎨 Customization

### Colors
Edit `src/app/globals.css` to change theme colors:
```css
:root {
  --primary: #4F46E5; /* Change brand color */
  --background: #F3F4F6; /* Change background */
}
```

### Charts
Modify chart colors in individual components:
```tsx
const COLORS = ['#4F46E5', '#10B981', '#F59E0B']
```

## 📝 Code Quality

### Comments
- Component purpose documented at top
- Complex logic explained inline
- Props interfaces with descriptions
- API endpoints documented

### Structure
- Consistent file organization
- Reusable components
- Separated concerns (UI, logic, data)
- Type-safe with TypeScript

### Best Practices
- Semantic HTML elements
- Accessibility features (ARIA labels, keyboard navigation)
- SEO optimized (meta tags, proper headings)
- Error boundaries for robustness

## 🔒 Security Considerations

- CORS configured for specific origins
- Environment variables for sensitive data
- Input sanitization on backend
- SQL injection prevention (JPA/Hibernate)

## 🚧 Future Enhancements

- [ ] Real-time updates with WebSockets
- [ ] Advanced filtering and sorting
- [ ] Data export (CSV, PDF)
- [ ] Date range pickers for all charts
- [ ] Dark mode toggle
- [ ] Multi-language support
- [ ] Push notifications
- [ ] Advanced analytics with ML predictions

## 📞 API Documentation

### Dashboard Endpoints

#### GET `/api/dashboard/metrics`
Returns overall dashboard metrics.

#### GET `/api/dashboard/revenue`
Returns revenue data points for chart.

#### GET `/api/dashboard/installments`
Returns monthly installment data.

#### GET `/api/dashboard/status-distribution`
Returns payment status breakdown.

#### GET `/api/dashboard/recent-transactions?limit=10`
Returns recent transactions with optional limit.

#### GET `/api/dashboard/payment-plans?status=Active&limit=10`
Returns payment plans with optional filters.

---

**Built with ❤️ using Next.js, TypeScript, and modern web technologies.**
