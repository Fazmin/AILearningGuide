import type { ModuleCardInfo } from "@app/module-sdk";

/**
 * Info-icon explanations for the six cards, keyed by the exact `label` of each <LabSurface>. Cards 1,
 * 2 and 5 draw published tables and figures; cards 3 and 4 are small models written for this lab, and
 * each entry says which is which. The figures quoted are pinned in quoted.test.ts.
 */
const cardInfo: ModuleCardInfo = {
  "Narrow, general, and how good": {
    title: "A map with two axes instead of one ladder",
    summary:
      "The Levels of AGI framework grades a system on two questions: how well it performs compared with people, and how wide a range of tasks it handles. Narrow AI, AGI and ASI become cells in that table, so a system can be superhuman at one task without being general at all.",
    whatYouSee: [
      "Rows are the table's six performance levels, from 0 (No AI) at the bottom to 5 (Superhuman) at the top. Columns are narrow (one task or a few) and general (a wide range of tasks).",
      "The ASI cell, level 5 general, has a dashed pink border. Level 2 general carries the paper's own label, Competent AGI. Hatched general cells are the ones the paper marks not yet achieved.",
      "Eight system chips above the table. The cell you place a chip in shows it; once all eight are placed, each chip gets a tick or a cross, and every cell shows the systems the table lists there.",
      "The panel beside the table repeats the selected cell's performance bar and the examples the table gives for it.",
    ],
    howItWorks: [
      "Every bar and example is copied from Table 1 of Morris et al. (ICML 2024). Level 2 is at least the 50th percentile of skilled adults, level 3 the 90th, level 4 the 99th, and level 5 outperforms 100% of humans.",
      "A placement is right when it matches the cell the table lists that system in. Show the table's answers fills in all eight.",
    ],
    controls: [
      "Click a chip to choose it, then click a cell. The next unplaced chip is chosen for you. Clear the table empties it; Show the table's answers fills it.",
    ],
    notice: [
      "The top row already has narrow entries: AlphaFold, AlphaZero and Stockfish each beat every person at one task.",
      "The general column is filled only at levels 0 and 1, where the paper puts ChatGPT, Bard, Llama 2 and Gemini as Emerging AGI.",
      "ASI is the one cell that needs both: superhuman and general.",
    ],
    limits: [
      "In this lab: the placements are the paper's judgments as of its publication. Newer systems are not on the table, and the lab does not grade them.",
      "In this lab: the paper also grades autonomy on a separate scale, from AI as a tool to AI as an agent, which this card leaves out.",
      "In general: percentile bars need a test for each task and a pool of skilled adults to compare against. For most real tasks neither exists, so placing a general system on this table is a judgment, not a measurement.",
    ],
  },

  "One profile, three verdicts": {
    title: "Whether it is AGI depends on how you read the scores",
    summary:
      "One published profile of scores, read three ways. The paper adds its ten areas into an AGI score; letting the weakest area decide, or counting only areas at adult level, gives very different answers from the same numbers. The chart's edge is one adult, so it has no room for anything beyond a person.",
    whatYouSee: [
      "A ten-sided chart, one spoke per area of thinking. The solid outline is a well-educated adult, worth 10 points per area; the dashed pink outline beyond it marks where a superhuman score would go, which this scale does not record.",
      "The filled shape is the chosen model's scores and the dashed shape the other model's. A hollow red dot marks an area scored 0.",
      "Three bars, one per rule, give how close to AGI the profile is under each. The formula row works out the chosen rule.",
    ],
    howItWorks: [
      "Scores are from Table 1 of Hendrycks et al. (2025). GPT-4: knowledge 8, reading and writing 6, maths 4, reasoning 0, working memory 2, memory storage 0, memory retrieval 4, seeing 0, hearing 0, speed 3. GPT-5: 9, 10, 10, 7, 4, 0, 4, 4, 6, 3.",
      "Add up the domains is the paper's AGI score: the sum out of 100, 27% for GPT-4 and 57% for GPT-5. Weakest domain decides is the lowest score out of 10. Count domains at adult level is the share of areas scoring the full 10.",
    ],
    controls: ["Model switches between GPT-4 and GPT-5. Rule picks which reading the formula row works out."],
    notice: [
      "GPT-5 reads 57%, 0% and 20% under the three rules. GPT-4 reads 27%, 0% and 0%.",
      "Long-term memory storage is 0 for both models, which is why the weakest-area rule gives 0% for both.",
      "From GPT-4 to GPT-5 the sum doubles, while the weakest-area reading does not move at all.",
    ],
    limits: [
      "In this lab: the scores are the paper's authors' own assessments, not tests the lab runs. Only the paper's sum is its definition; the other two rules are readings the lab adds to show that the definition matters.",
      "In this lab: two models only. The paper's method and scores may be revised.",
      "In general: any test that caps at human level cannot tell AGI from ASI, and any single number hides which areas are weak. Other groups define AGI by percentile, by economic value or by autonomy instead.",
    ],
  },

  "Faster, more, or better": {
    title: "What speed and copies buy, and what they cannot",
    summary:
      "A project measured in expert-years, done by many copies of a mind that thinks faster than a person. Thinking that can be split across copies shrinks with both; thinking that must happen in order shrinks only with speed; work that waits on the outside world does not shrink at all. That last share sets a floor no mind can beat.",
    whatYouSee: [
      "A time ruler on which each labelled step is far longer than the last, from 1 second to 100 years. The grey pin is one expert alone; the blue pin is the project with your settings.",
      "Below it, one bar split into three shares of the total: waiting on the world, thinking in order, and thinking split across copies.",
      "The formula row, and tiles for one expert alone, your settings, where most of the time goes, and the fastest the project could ever finish with these shares.",
    ],
    howItWorks: [
      "Calendar time = work × [world + (1 − world) × (in order ÷ speed + (1 − in order) ÷ (speed × copies))].",
      "Fastest possible here uses a speed and a copy count of a trillion each, so it is in effect the world share alone.",
      "This is a model written for this lab, a version of Amdahl's law with an extra share that waits on the world. It has no cost for coordinating copies.",
    ],
    controls: [
      "Project (1 to 1,000 expert-years), Speed and Copies (1× to 1 million×), Must happen in order and Waits on the world (0% to 50%).",
      "Comparison worth running: 100 expert-years, Speed 1,000×, Copies 1 million×, then Waits on the world at 0% and at 10%.",
    ],
    notice: [
      "With nothing waiting on the world, a 100 expert-year project at 1,000× speed and a million copies takes about 3 seconds. With 10% waiting on the world, it takes 10 years.",
      "With everything in order, extra copies do nothing at all; only speed helps.",
      "There is no control for thinking better. A better idea is not the same as a faster one, and the arithmetic cannot represent it.",
    ],
    limits: [
      "In this lab: the shares and multipliers are inputs, not estimates of any real system. Real copies need to coordinate, and that cost is left out, which flatters the copies.",
      "In this lab: the world share is treated as fixed. In reality better ideas can also redesign experiments to need less waiting.",
      "In general: nobody knows how much of research waits on the physical world, or whether a mind that thinks better, not only faster, could get around those waits. That is one of the main disagreements about how quickly ASI could change things.",
    ],
  },

  "A loop that feeds itself": {
    title: "The intelligence explosion as a model you can run",
    summary:
      "A toy model of AI doing AI research. Once AI matches people at research, it takes over the work it can do, so research speeds up as AI improves. Whether that loop runs away depends on two numbers nobody has measured: how much each doubling of research effort improves AI, and how much of the work only people can do.",
    whatYouSee: [
      "Three boxes for the loop at the year you pick: AI research ability (1× is today's human researchers), research speed, and the returns setting.",
      "A plot of AI research ability against years in the toy, on a scale where each step up is ten times more. Dotted lines mark 1×, matching people, and 100×, the lab's stand-in for far beyond people. The shaded band is the climb between them.",
      "A dashed curve is your previous run. A marker at the top edge shows where a curve has left the chart, past a million times.",
    ],
    howItWorks: [
      "Research speed R = 1 ÷ (b + (1 − b) ÷ max(1, A)), where A is AI ability and b is the share only people can do. Below 1×, people do all the research and R is 1.",
      "Progress follows d(ln A) / dt = 0.15 × R^r ÷ A, solved step by step from A = 0.1 in year 0. With no bottleneck and A above 1, that is 0.15 × A^(r − 1).",
      "So returns r above 1 make each doubling arrive sooner, r = 1 gives a steady doubling time, and r below 1 makes growth slow down. With b above 0, R can never pass 1 ÷ b.",
    ],
    controls: [
      "Returns to research (0.5 to 3), Only people can do (0% to 20%), and Year, which moves the marker and the three boxes.",
      "Comparison worth running: returns 1.5 with 0% and then 20%; then returns 3 with 20%.",
    ],
    notice: [
      "Every run matches people in year 6, because before that people do all the research.",
      "With no bottleneck: returns 1 reach 100× in year 36.7, returns 1.5 in year 18.0 and returns 3 in year 9.3. Returns below 1 never reach 100× in 40 years.",
      "At returns 1.5, a 1% human-only share barely changes the date for 100× (year 19.8 instead of 18.0) but stops the curve leaving the chart; 20% stops it reaching 100× at all.",
    ],
    limits: [
      "In this lab: the formula, the starting point and the 0.15 yearly gain are made up so the shapes are easy to see. The years mean nothing, and neither do 1× and 100× as measures of intelligence.",
      "In this lab: research ability is one number. Real research needs compute, data and experiments, each of which could be its own bottleneck.",
      "In general: the returns to AI research and the share of it only people can do have not been measured, and published estimates disagree. The model shows how the argument works, not whether its conclusion is true.",
    ],
  },

  "Reading forecasts": {
    title: "Two kinds of forecast, and what moves each",
    summary:
      "Forecasts about AGI come in two kinds. Surveys ask experts, and their answers move with the wording of the question and from one year to the next. Trend lines carry a measurement forward, and their answers move with the assumed rate. The card shows one of each.",
    whatYouSee: [
      "A timeline from 2020 to 2130 with two questions from the same survey. A hollow dot is the year given a 10% chance and a filled dot the year given a 50% chance.",
      "A plot of task length against year, on a scale where each step up is much longer. The shaded band is the measured stretch, 2019 to early 2025, drawn as the fitted trend; the dashed line carries it forward at your doubling time, and the faint dashed lines show the other doubling times.",
      "Tiles for the carried-forward length, the length at the measured doubling time, the year, and the survey's 50% year.",
    ],
    howItWorks: [
      "Survey: Grace et al. (2024), 2,778 published AI researchers. Machines beating people at every task: 10% by 2027, 50% by 2047, 13 years earlier than the previous survey's 2060. Every occupation fully automatable: 10% by 2037, 50% by 2116.",
      "Trend: Kwa et al. (2025) report a 50% time horizon of about 50 minutes for early-2025 frontier models, doubling about every 7 months since 2019. The card computes 50 minutes × 2^(12 × years ÷ doubling months).",
      "Task lengths are said in working time: 8-hour days, 40-hour weeks, 167-hour months and 2,000-hour years.",
    ],
    controls: [
      "Survey switches between late 2023 and 2022. Years after early 2025 moves the cursor; Assumed doubling time picks 4, 7, 12 or 24 months.",
    ],
    notice: [
      "The same respondents put the two questions 69 years apart.",
      "At 7 months, five years on gives about 317 hours, 1.9 working months, matching the paper's own projection of month-long tasks within five years. At 12 months it gives about 27 hours; at 4 months, about 27,000.",
      "Nothing to the right of the shaded band is a measurement.",
    ],
    limits: [
      "In this lab: the fitted trend is drawn from the paper's start point and doubling time, not from its individual data points. The 2022 survey's figure for every occupation is not shown.",
      "In this lab: the paper's time horizon is for software tasks in its test set. It adds that its projection holds only if the results generalize to real-world software tasks.",
      "In general: survey answers depend on wording and on who answers, and trend lines can bend either way. Neither is a measurement of when AGI or ASI will arrive.",
    ],
  },

  "Challenge: sort the claims": {
    title: "Four kinds of claim, four kinds of trust",
    summary:
      "Eight claims about AGI and ASI, each from an earlier card, to sort as a measurement, a definition, an argument or a forecast. Each kind is checked differently: measurements by evidence, definitions by usefulness, arguments by their assumptions, and forecasts by waiting.",
    whatYouSee: [
      "Eight numbered claims, each with four buttons. Once you choose, the claim's edge turns green or red and a short reason appears under it.",
      "A meter counting claims sorted correctly, out of 8.",
    ],
    howItWorks: [
      "Each claim has one intended kind, two of each: Measured for Stockfish and the task-length trend, Definition for the two definitions of AGI and ASI, Argument for the feedback loop and the orthogonality thesis, Forecast for 2047 and for superintelligence within ten years of AGI.",
      "Answers are stored, so you can change any of them and leave and come back.",
    ],
    controls: ["Measured, Definition, Argument and Forecast under each claim; Start over clears every answer."],
    notice: [
      "A claim can lean on another kind: the 2047 forecast comes from a measured survey, and the trend claim is measured even though carrying it forward is not.",
      "None of the eight is a measurement of AGI or ASI themselves.",
    ],
    limits: [
      "In this lab: each claim is worded so that one kind fits best. Real statements often mix kinds in one sentence.",
      "In general: sorting a claim does not settle whether it is true. A measurement can be flawed, a definition can be unhelpful, an argument can be valid with false assumptions, and a forecast can turn out right.",
    ],
  },
};

export default cardInfo;
