# مؤشرات Pine Script

نسخ Pine من سكانرات المستودع، لتشوف نفس الإشارات على شارت TradingView وتضع عليها تنبيهات.
كل الفحوص على شموع **مغلقة** فقط (بلا إعادة رسم)، والأرقام الافتراضية نفس أرقام السكانر.

| الملف | السكانر الأصلي | الشارت المطلوب |
|---|---|---|
| `macd_cascade.pine` | `macd_cascade_scanner.py` + `macd_slow_scanner.py` | سريعة: 15 دقيقة · بطيئة: ساعة (اختر النسخة من الإعدادات) |
| `ema_cross_weekly.pine` | `ema_cross_scanner.py` | أي فاصل (المتوسطات أسبوعية دائماً) |
| `ma100_200_breakout.pine` | `ma100_200_scanner.py` | يومي |
| `spy_orb_0dte.pine` | `spy0dte/spy_orb_0dte.py` | SPY على 5 دقائق، الجلسة العادية فقط |

## التركيب يدوياً

1. افتح Pine Editor في TradingView.
2. الصق محتوى الملف ← **Add to chart**.
3. للتنبيه: Alerts ← Create alert ← اختر المؤشر ← **Any alert() function call**
   (أو اختر أحد شروط `alertcondition` بالاسم).

## التركيب عبر tradingview-mcp

على جهازك، مع TradingView Desktop مفتوح بمنفذ التصحيح
([الإعداد](https://github.com/tradesdontlie/tradingview-mcp)):

```bash
tv pine check   -f pine/macd_cascade.pine   # فحص الترجمة على خادم TradingView (بلا شارت)
tv pine set     -f pine/macd_cascade.pine   # ضع الكود في المحرر
tv pine compile                             # أضفه للشارت
tv pine errors                              # أخطاء الترجمة إن وُجدت
```

أو من Claude Code: «افتح NVDA على 15 دقيقة وأضف pine/macd_cascade.pine».

## فروق عن السكانر

- **السيولة والقوائم**: المؤشر يعمل على الرمز المفتوح فقط، فلا فلتر سيولة دولاري ولا قائمة رموز.
  فلتر `ma100_200` السيولي (3M$) موجود لأنه مبني على شمعة واحدة.
- **MACD Cascade**: شمعات 4H من TradingView نفسها (للأسهم تبدأ من افتتاح الجلسة، وللعملات على حدود UTC)
  — نفس تقسيم السكانر تقريباً. القيم قد تختلف قليلاً عن yfinance بسبب اختلاف مصدر البيانات.
- **SPY ORB**: المؤشر يرسم النطاق والإشارة وسترايك ATM ووقت الخروج فقط.
  سعر العقد والتقييم الورقي (Polygon) يبقيان في السكانر.
- **EMA الأسبوعي**: الإشارة تظهر على أول شمعة بعد إغلاق الأسبوع، مثل السكانر الذي يعمل بعد إغلاق الجمعة.
- `smc_breakout_scanner.py` غير منقول: طبقته الأولى فلتر Finviz، ولا يمكن تنفيذه داخل Pine.
