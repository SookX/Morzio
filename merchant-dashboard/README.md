# Morzio Merchant Dashboard

A modern, interactive, and fully responsive Next.js dashboard for visualizing payment plans and installments. Built with cutting-edge technologies and stunning animations.

![Dashboard Preview](./preview.png)

## ✨ Features

### 🎨 Design
- **Clean & Minimalist UI**: Elegant interface with subtle gradients, shadows, and rounded elements
- **Responsive Design**: Fully optimized for desktop, tablet, and mobile devices
- **Modern Color Palette**: Curated indigo-based color scheme with smooth transitions
- **Dark Mode Ready**: Color system designed for easy dark mode implementation

### 📊 Data Visualization
- **Interactive Charts**: 
  - Line chart for revenue trends over time
  - Bar chart for monthly installment tracking
  - Donut chart for payment status distribution
- **Real-time Metrics**: Key performance indicators displayed in animated cards
- **Transaction History**: Recent payment activities with status indicators
- **Payment Plan Cards**: Visual progress tracking for each installment plan

### 🎭 Animations & UX
- **Framer Motion**: Smooth page transitions and micro-interactions
- **Hover Effects**: Interactive states on all clickable elements
- **Staggered Animations**: Sequential loading for a polished feel
- **Progress Bars**: Animated installment completion tracking

### 🏗️ Technical Stack

#### Core
- **Next.js 16.0.5** (App Router)
- **React 19.2.0**
- **TypeScript** for type safety
- **Tailwind CSS 4.x** for styling

#### Libraries
- **Recharts** - Beautiful, composable charts
- **Framer Motion** - Production-ready animations
- **Lucide React** - Modern icon library
- **clsx & tailwind-merge** - Utility class management

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   cd merchant-dashboard
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Run the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

### Build for Production

```bash
npm run build
npm start
```

## 📁 Project Structure

```
merchant-dashboard/
├── src/
│   ├── app/                    # Next.js app directory
│   │   ├── layout.tsx         # Root layout with Sidebar & Header
│   │   ├── page.tsx           # Main dashboard page
│   │   └── globals.css        # Global styles & CSS variables
│   ├── components/
│   │   ├── dashboard/         # Dashboard-specific components
│   │   │   ├── MetricsCards.tsx
│   │   │   ├── RevenueChart.tsx
│   │   │   ├── MonthlyInstallmentsChart.tsx
│   │   │   ├── PaymentStatusChart.tsx
│   │   │   ├── RecentTransactions.tsx
│   │   │   └── PaymentPlanCard.tsx
│   │   ├── layout/            # Layout components
│   │   │   ├── Sidebar.tsx
│   │   │   └── Header.tsx
│   │   └── ui/                # Reusable UI components
│   │       ├── button.tsx
│   │       └── card.tsx
│   └── lib/
│       └── utils.ts           # Utility functions
├── public/                    # Static assets
├── package.json
└── README.md
```

## 🎨 Key Components

### MetricsCards
Displays key performance indicators with icons and trend information:
- Total Revenue
- Pending Installments
- Active Plans
- Completed Payments

### Charts
- **RevenueChart**: Line chart showing revenue trends over 7 months
- **MonthlyInstallmentsChart**: Bar chart for monthly collection tracking
- **PaymentStatusChart**: Donut chart for status distribution

### PaymentPlanCard
Individual payment plan cards featuring:
- Customer information
- Progress tracking with animated bars
- Status badges (Active, Completed, Overdue)
- Next due date indicators
- Hover interactions

### Layout
- **Sidebar**: Responsive navigation with mobile menu support
- **Header**: Search functionality and notification center

## 🔧 Customization

### Colors
Edit CSS variables in `src/app/globals.css`:
```css
:root {
  --primary: #111827;
  --accent: #4F46E5;
  /* ... */
}
```

### Mock Data
Currently using dummy data. To integrate with a real backend:

1. Create API routes in `src/app/api/`
2. Update components to fetch from your endpoints
3. Add loading states and error handling

Example:
```typescript
// In your component
const { data } = await fetch('/api/payment-plans')
```

## 📱 Responsive Breakpoints

- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

## 🎯 Future Enhancements

- [ ] Add filtering and date range selectors
- [ ] Implement real-time data updates
- [ ] Add export functionality (PDF, CSV)
- [ ] Create detailed analytics page
- [ ] Add customer management interface
- [ ] Implement authentication
- [ ] Add dark mode toggle

## 🤝 Contributing

This is a demo project created for the Morzio payment platform. Feel free to fork and customize for your own needs.

## 📄 License

MIT License - feel free to use this project for your own purposes.

## 🙏 Acknowledgments

- Design inspired by modern fintech dashboards
- Built with love using Next.js and React
- Icons by Lucide
- Charts powered by Recharts

---

**Made with ❤️ for Morzio**
