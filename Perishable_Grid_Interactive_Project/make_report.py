from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parent
PDF=BASE/'Perishable_Grid_Hackathon_Brief.pdf'

# Unicode-capable fonts for ₹ and common punctuation.
font_regular='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font_bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
pdfmetrics.registerFont(TTFont('DejaVu',font_regular))
pdfmetrics.registerFont(TTFont('DejaVu-Bold',font_bold))

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom', parent=styles['Title'], fontName='DejaVu-Bold', fontSize=25, leading=30, textColor=colors.HexColor('#123047'), alignment=TA_LEFT, spaceAfter=5))
styles.add(ParagraphStyle(name='SubCustom', parent=styles['Normal'], fontName='DejaVu', fontSize=11, leading=16, textColor=colors.HexColor('#506475'), spaceAfter=10))
styles.add(ParagraphStyle(name='H1Custom', parent=styles['Heading1'], fontName='DejaVu-Bold', fontSize=16, leading=20, textColor=colors.HexColor('#123047'), spaceBefore=3, spaceAfter=7))
styles.add(ParagraphStyle(name='H2Custom', parent=styles['Heading2'], fontName='DejaVu-Bold', fontSize=11, leading=14, textColor=colors.HexColor('#176B73'), spaceBefore=6, spaceAfter=4))
styles.add(ParagraphStyle(name='BodyCustom', parent=styles['BodyText'], fontName='DejaVu', fontSize=8.6, leading=12, textColor=colors.HexColor('#263746'), spaceAfter=5))
styles.add(ParagraphStyle(name='SmallCustom', parent=styles['BodyText'], fontName='DejaVu', fontSize=7.1, leading=9.2, textColor=colors.HexColor('#425466'), spaceAfter=3))
styles.add(ParagraphStyle(name='Callout', parent=styles['BodyText'], fontName='DejaVu-Bold', fontSize=11, leading=15, textColor=colors.HexColor('#123047'), backColor=colors.HexColor('#EAF3F4'), borderColor=colors.HexColor('#A7CED0'), borderWidth=0.7, borderPadding=8, spaceBefore=5, spaceAfter=8))
styles.add(ParagraphStyle(name='Tiny', parent=styles['BodyText'], fontName='DejaVu', fontSize=6.5, leading=8, textColor=colors.HexColor('#52616B')))


def P(txt, style='BodyCustom'):
    return Paragraph(txt, styles[style])

def footer(canvas, doc):
    canvas.saveState()
    w,h=A4
    canvas.setStrokeColor(colors.HexColor('#D7E0E5'))
    canvas.line(16*mm, 13*mm, w-16*mm, 13*mm)
    canvas.setFont('DejaVu',7)
    canvas.setFillColor(colors.HexColor('#607482'))
    canvas.drawString(16*mm,8*mm,'THE PERISHABLE GRID | HACKATHON PROTOTYPE')
    canvas.drawRightString(w-16*mm,8*mm,f'{doc.page}')
    canvas.restoreState()

story=[]

# PAGE 1
story += [P('THE PERISHABLE GRID','TitleCustom'),
          P('Physical energy trading with battery-health and customer-SLA-aware dispatch','SubCustom'),
          P('THE IDEA','H2Custom'),
          P('The operator is balancing three scarce resources at once: <b>grid energy, remaining battery life and customer time</b>. A normal dispatch rule tends to buy power, cycle batteries and queue customers using separate decisions. The Perishable Grid puts those choices in one hourly decision loop.','BodyCustom'),
          P('The distinctive mechanism is a <b>triple marginal-cost comparison</b>: before serving the next unit of demand, compare (1) the cost of buying one more kWh, (2) the marginal wear cost of using a battery pack, and (3) the cost of shifting a flexible customer to another time or nearby hub.','Callout'),
          P('WHAT THE FOUR-HOUR PROTOTYPE ACTUALLY BUILDS','H1Custom')]
arch_data=[
    [P('<b>1. Forecast</b><br/>One-day seasonal baseline for demand, renewable output and market price.','SmallCustom'),P('<b>2. Dispatch</b><br/>Choose grid source; charge/discharge storage under power, SoC and reserve limits.','SmallCustom')],
    [P('<b>3. Battery health</b><br/>Assign high-SoH packs to Rapid service; suitable ageing packs to shallow stationary use.','SmallCustom'),P('<b>4. SLA shaping</b><br/>Shift a defined share of flexible evening load into solar-rich hours, within the tariff cap.','SmallCustom')]
]
t=Table(arch_data,colWidths=[84*mm,84*mm],rowHeights=[25*mm,25*mm])
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#F2F7F8')),
    ('BOX',(0,0),(-1,-1),0.8,colors.HexColor('#C9D8DE')),
    ('INNERGRID',(0,0),(-1,-1),0.6,colors.HexColor('#D8E2E6')),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9)
]))
story += [t,Spacer(1,7),P('WHAT IS IN SCOPE - AND WHAT IS NOT','H2Custom'),
          P('<b>Implemented:</b> five-cluster synthetic simulator, seasonal-naive forecast, four dispatch policies, explicit 70/30 SoH-tier allocation, 5% reserve, grid and inverter caps, and same-metro time shifting of flexible demand.','BodyCustom'),
          P('<b>Not implemented:</b> trained TFT/XGBoost, production MILP, live IEX bids, VPP market participation, physical battery repurposing, inter-hub pack transport or a DC-coupled hardware design. These remain extensions, not claimed capabilities.','BodyCustom'),
          P('Why the trading-desk framing is credible: IEX DAM is a physical next-day market using double-sided closed auctions, while RTM is a separate market with half-hourly auction sessions and delivery one hour after closure [1,2]. The prototype uses synthetic price inputs and does not connect to either market.','SmallCustom')]

# PAGE 2
story += [PageBreak(),P('MODEL AND ASSUMPTIONS','H1Custom'),
          P('The prototype is deliberately smaller than the final architecture. It uses a transparent heuristic so a judge can inspect every decision. A production system can replace the heuristic with rolling-horizon MILP/MPC once real asset and market data are available.','BodyCustom'),
          P('1. BATTERY ASSIGNMENT IS EXPLICIT','H2Custom'),
          P('For each pack <i>i</i>, define binary assignments <i>z</i><sub>i,R</sub>, <i>z</i><sub>i,S</sub> and <i>z</i><sub>i,E</sub> for Rapid service, second-life stationary use and end-of-life/recycling:','BodyCustom'),
          P('<b>z</b><sub>i,R</sub> + <b>z</b><sub>i,S</sub> + <b>z</b><sub>i,E</sub> = 1; &nbsp; z<sub>i,R</sub>, z<sub>i,S</sub>, z<sub>i,E</sub> ∈ {0,1}.','Callout'),
          P('Prototype eligibility assumptions: a pack with SoH ≥ 85% and a passed safety screen may enter Rapid or stationary service; a pack with 70% ≤ SoH < 85% and a passed safety screen may enter stationary service only; a pack below 70% or failing safety checks is excluded from dispatch. These cutoffs are test assumptions, not universal battery standards.','BodyCustom'),
          P('2. ENERGY AND STORAGE CONSTRAINTS','H2Custom'),
          P('For each cluster and hour, renewable energy + grid imports + battery discharge = customer load + battery charge + losses + permitted export. Grid import is capped at 38 MW per cluster in this prototype. The assumed battery inverter limit is 30 MW per cluster. The case gives 600 MWh total storage; the prototype assumes an equal 120 MWh per cluster split.','BodyCustom'),
          P('SoC<sub>t+1</sub> = SoC<sub>t</sub> + η<sub>ch</sub>P<sub>ch</sub>Δt − P<sub>dis</sub>Δt / η<sub>dis</sub>','Callout'),
          P('The simulation uses η<sub>ch</sub> = η<sub>dis</sub> = 95%, a 5% reserve, a 50% starting SoC, and a simplified 97.5% delivery efficiency. Charging and discharging power, storage capacity and reserve are checked each interval.','BodyCustom'),
          P('3. TRIPLE MARGINAL-COST RULE','H2Custom'),
          P('For the next unit of served demand, compare: <b>C<sub>grid</sub></b> = market/DISCOM cost; <b>C<sub>wear,i</sub></b> = marginal battery-life cost for pack i; and <b>C<sub>shift,k</sub></b> = incentive + customer delay/detour cost for segment k. Use the lowest-cost feasible action while satisfying service priority, grid and reserve constraints.','BodyCustom'),
          P('The future MILP objective is expected customer revenue minus procurement, degradation, incentive and unserved-demand costs, with an optional CVaR penalty for downside scenarios. VPP revenue is excluded from the base case. Tariffs can be chosen from discrete values around ₹14/kWh to keep the price-demand term linear.','BodyCustom'),
          P('4. ASSUMPTIONS USED IN THE RUN','H2Custom')]
assumptions=[
    ['Parameter','Assumed value for demo'],
    ['Battery per metro','120 MWh; 30 MW inverter'],
    ['Grid import cap','38 MW per metro'],
    ['SoH mix','70% Rapid at 90% SoH; 30% stationary at 80% SoH'],
    ['Energy efficiencies','95% charge; 95% discharge; 97.5% delivery'],
    ['Prices','DISCOM ₹8.50/kWh; renewable ₹3.50/kWh; customer ₹14/kWh'],
    ['IEX test price','Synthetic ₹3.50–₹11.50/kWh; not historical exchange data'],
    ['Demand shifting',"40% flexible segment; shift 15% of that segment\'s peak demand; 10% incentive"],
]
t=Table([[P(str(c),'Tiny') for c in row] for row in assumptions],colWidths=[50*mm,118*mm])
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF1F4')),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#C9D5DA')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
story.append(t)

# PAGE 3
story += [PageBreak(),P('SIMULATION: WHAT THE MODEL ACTUALLY DID','H1Custom'),
          P('Test design: 14 synthetic test days across five metro clusters, preceded by one warm-up day for the forecast. Demand, renewable output and IEX-like prices are generated with a fixed random seed. Two heatwave days increase demand by 18%; two monsoon days reduce solar output by 45% and include an evening price spike; two surplus days increase solar generation by 22%. These are stress-test assumptions, not weather records.','BodyCustom'),
          P('The forecast is a one-day seasonal-naive baseline: the previous day\'s value for the same cluster and hour. The dispatch logic uses forecast price and expected supply gap to schedule storage, then balances actual synthetic demand in the operating hour.','BodyCustom')]
summary=pd.read_csv(BASE/'summary_results.csv').set_index('policy')
order=['baseline','forecast','health','full']
labels=['Rule-based','Price-aware','Health-aware','Health + SLA']
rows=[[P('<b>Policy</b>','Tiny'),P('<b>Reliability</b>','Tiny'),P('<b>Gross contribution*</b>','Tiny'),P('<b>Wear proxy</b>','Tiny')]]
for key,label in zip(order,labels):
    r=summary.loc[key]
    rows.append([P(label,'Tiny'),P(f'{r.reliability_pct:.2f}%','Tiny'),P(f'₹{r.gross_contribution_rs_lakh/100:.2f} cr','Tiny'),P(f'₹{r.wear_cost_rs_lakh:.2f} lakh','Tiny')])
t=Table(rows,colWidths=[46*mm,35*mm,47*mm,40*mm])
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF1F4')),('GRID',(0,0),(-1,-1),0.45,colors.HexColor('#C9D5DA')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
story += [t,Spacer(1,3),P('*Gross contribution = simulated customer revenue less simulated grid/renewable procurement and wear-cost proxy. It excludes fixed O&M, capex, taxes and depreciation; it is not EBITDA.','Tiny'),Spacer(1,5),
          Image(str(BASE/'chart_reliability.png'),width=168*mm,height=94*mm),
          P('The full policy reaches 87.58% in this synthetic run, still 2.12 percentage points below the case target of 89.7%. The synthetic rule baseline is 81.70%, not the case-reported 78%; the simulation has not been calibrated to operator data.','SmallCustom')]

# PAGE 4
story += [PageBreak(),P('WHAT THE RESULTS SUPPORT - AND WHAT THEY DO NOT','H1Custom'),
          Image(str(BASE/'chart_contribution.png'),width=168*mm,height=94*mm),
          Image(str(BASE/'chart_wear.png'),width=168*mm,height=94*mm),
          P('Reading the ablation: adding health-aware dispatch lowers the wear-cost proxy by about 8.6% versus price-aware dispatch, with similar reliability. Adding SLA shifting then improves reliability by about 1.1 percentage points versus health-aware dispatch and lowers the wear proxy by a further 6.6%. The rule-based policy has low wear partly because it under-serves demand; low wear alone is not a success.','SmallCustom'),
          P('A representative monsoon / price-spike demand-versus-delivery curve is included as chart_stress_day.png in the accompanying code bundle.','SmallCustom'),
          P('LIMITATIONS TO STATE IN THE PITCH','H2Custom'),
          P('This prototype is a dispatch heuristic, not a solved MILP. It does not train TFT/XGBoost, implement live bidding, calculate VPP revenue, move physical packs between hubs or calculate EBITDA. Its simplified loss proxy is not comparable with the case\'s 10% loss figure because the actual metering boundary is not supplied. The 8% loss target therefore remains unverified.','SmallCustom'),
          P('WHY THE INDUSTRY REFERENCES MATTER','H2Custom'),
          P('Sources: IEX DAM/RTM; Tesla Autobidder; Fluence Mosaic; Octopus smart charging; CERC Ancillary Services Regulations (2022); NITI Aayog on second-life batteries; and the 2024 battery-degradation paper in Wiley\'s <i>Energy Storage</i>. Full URLs are in README.','Tiny')]

doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=14*mm,bottomMargin=18*mm,title='The Perishable Grid - Hackathon Brief',author='')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(PDF)
