import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { hydrateState, initialState } from "./state";

const definition: ModuleDefinition = {
  id: "module-62-agi-superintelligence",
  slug: "agi-superintelligence",
  title: "AGI & superintelligence",
  group: "frontiers",
  order: 39,
  icon: "Telescope",
  accent: "#8a4fbf",
  prerequisites: ["module-29-what-ai-is", "module-10-scaling-laws", "module-12-post-training"],
  estimatedMinutes: 22,
  steps: ["Place the systems", "One profile, three verdicts", "Faster or more", "The feedback loop", "Reading forecasts", "Sort the claims"],
  stepInstructions: [
    "Answer Predict first. Pick a system, click the cell where you think the Levels of AGI table puts it, and repeat for all eight. Then click the ASI corner and read what the table says.",
    "Answer Predict first. With GPT-5 selected, switch the Rule between Add up the domains, Weakest domain decides and Count domains at adult level, then switch Model to GPT-4 and compare.",
    "Answer Predict first. Raise Speed and Copies and watch the calendar time fall, then raise Waits on the world to 10% and compare Fastest possible here with what you had.",
    "Answer Predict first. Step Returns to research from 0.5 up to 3 and watch the curve, then set Only people can do to 20% and compare with the dashed curve of your previous run.",
    "Answer Predict first. Switch Survey between Late 2023 and 2022, then drag Years after early 2025 to 5 and change Assumed doubling time from 7 mo to 4 mo and 12 mo.",
    "Sort all eight claims into Measured, Definition, Argument or Forecast, and read the reason under each one.",
  ],
  stateVersion: 1,
  tagline:
    "Place real systems on a two-axis map of AI levels, see one score profile give three verdicts on “is it AGI?”, compute what faster and more copies buy, run a toy intelligence explosion, and sort the claims people make about superintelligence.",
  objectives: [
    "Place narrow AI, AGI and ASI on two separate axes, performance and generality, and say why a narrow superhuman system is not ASI",
    "Show that whether a system counts as AGI depends on the definition, by reading one jagged profile under three rules",
    "Compute how speed, copies and work that cannot be sped up limit what a faster mind could do, and say which kind of superiority the arithmetic leaves out",
    "Explain the intelligence-explosion argument as a feedback loop, and predict how returns to research and a human-only bottleneck change its shape",
    "Tell measurements, definitions, arguments and forecasts apart in claims about AGI and ASI, and say how the wording of a question and an assumed trend move a forecast",
  ],
  glossary: [
    {
      term: "Narrow AI",
      definition:
        "A system built for one task or a small set of tasks, such as playing chess or predicting protein shapes. It can be superhuman at that task and useless at everything else.",
    },
    {
      term: "Artificial general intelligence (AGI)",
      definition:
        "A system that can do a wide range of thinking tasks at least as well as people. Every group defines the bar differently, by percentile, by test areas or by economic value, so whether a system counts depends on whose definition is used.",
    },
    {
      term: "Artificial superintelligence (ASI)",
      definition:
        "A system much smarter than the best humans in practically every field. In the Levels of AGI table it is the top of the general column: superhuman and general at once. None exists, and it is not a measured property.",
    },
    {
      term: "Levels of AGI",
      definition:
        "A framework from Google DeepMind researchers that grades AI on performance (from no AI to superhuman) and generality (narrow or general) separately. It puts AGI and ASI in named cells rather than on one ladder.",
    },
    {
      term: "Jagged profile",
      definition:
        "A pattern of abilities that is strong in some areas and weak in others. Current AI systems are described this way, which is why one average score can hide an area where a system scores zero.",
    },
    {
      term: "Human-level machine intelligence",
      definition:
        "A survey phrase for machines that, unaided, can do every task better and more cheaply than people. It is one of several wordings of AGI, and the year experts give for it changes with the wording.",
    },
    {
      term: "Transformative AI",
      definition:
        "AI that changes the economy and society about as much as the industrial revolution did. The term is about impact, not about how smart a system is, so it could arrive before or after AGI.",
    },
    {
      term: "Intelligence explosion",
      definition:
        "The argument that an AI able to improve AI could start a loop in which each improvement speeds up the next, so ability races far beyond human. It depends on assumptions about returns and bottlenecks that nobody has measured.",
    },
    {
      term: "Recursive self-improvement",
      definition:
        "An AI improving its own design, or the design of its successors, which then improve further. It is the step that closes the intelligence-explosion loop.",
    },
    {
      term: "Takeoff",
      definition:
        "The stretch between AI roughly matching people and AI far beyond them. A fast takeoff is a short stretch and a slow takeoff a long one; how long it would be is an open question, not a measurement.",
    },
    {
      term: "Returns to research",
      definition:
        "How much more capable AI gets when research effort doubles. In the lab's loop, returns above 1 make each doubling arrive sooner than the last, returns of 1 give steady doubling, and returns below 1 make progress slow down.",
    },
    {
      term: "Bottleneck",
      definition:
        "A part of a process that cannot be sped up and so limits the whole, such as experiments in the physical world or work only people can do. Economists use it to argue that growth is held back by what is essential and hard to improve.",
    },
    {
      term: "Technological singularity",
      definition:
        "Vernor Vinge's 1993 name for the point where superhuman intelligence arrives and change becomes too fast for people to predict. It is a forecast and a metaphor, not a measured event.",
    },
    {
      term: "Orthogonality thesis",
      definition:
        "Nick Bostrom's argument that intelligence and goals are independent: a system could be extremely capable and still pursue almost any goal. It is why capability alone says nothing about whether a system is safe.",
    },
    {
      term: "Task time horizon",
      definition:
        "The length of a task, timed by how long it takes a person, that an AI system finishes half the time. METR measured it doubling about every seven months from 2019 to early 2025; carrying that forward is an assumption.",
    },
  ],
  references: [
    {
      authors: "Meredith Ringel Morris, Jascha Sohl-Dickstein, Noah Fiedel, et al.",
      title: "Levels of AGI for Operationalizing Progress on the Path to AGI",
      source: "Proceedings of the 41st International Conference on Machine Learning (ICML 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2311.02462",
      note: "The source of card 1. Its Table 1 grades AI by performance and generality, lists example systems in each cell, calls level 2 general Competent AGI, and names level 5 general Artificial Superintelligence, not yet achieved.",
    },
    {
      authors: "Dan Hendrycks, Dawn Song, Christian Szegedy, et al.",
      title: "A Definition of AGI",
      source: "arXiv:2510.18212",
      year: 2025,
      url: "https://arxiv.org/abs/2510.18212",
      note: "The source of card 2. It scores ten areas of thinking against a well-educated adult, reports GPT-4 at 27% and GPT-5 at 57%, and describes the resulting profile as highly jagged. The per-area scores on the card are from its Table 1.",
    },
    {
      authors: "Nick Bostrom",
      title: "How Long Before Superintelligence?",
      source: "International Journal of Futures Studies, vol. 2",
      year: 1998,
      url: "https://nickbostrom.com/superintelligence",
      note: "An early, widely quoted definition: an intellect much smarter than the best human brains in practically every field. The glossary's definition of ASI follows it.",
    },
    {
      authors: "Nick Bostrom",
      title: "Superintelligence: Paths, Dangers, Strategies",
      source: "Oxford University Press",
      year: 2014,
      url: "https://en.wikipedia.org/wiki/Superintelligence:_Paths,_Dangers,_Strategies",
      note: "The book that brought the idea to a wide audience, defining superintelligence as an intellect that greatly exceeds human cognitive performance in virtually all domains of interest. The link is an overview page; the book itself is not free to read.",
    },
    {
      authors: "Nick Bostrom",
      title: "The Superintelligent Will: Motivation and Instrumental Rationality in Advanced Artificial Agents",
      source: "Minds and Machines 22(2), 71–85",
      year: 2012,
      url: "https://doi.org/10.1007/s11023-012-9281-3",
      note: "States the orthogonality thesis, that intelligence and final goals are independent, which is one of the arguments sorted on the challenge card.",
    },
    {
      authors: "Irving John Good",
      title: "Speculations Concerning the First Ultraintelligent Machine",
      source: "Advances in Computers 6, 31–88",
      year: 1965,
      url: "https://www.semanticscholar.org/paper/d7d9d643a378b6fd69fff63d113f4eae1983adc8",
      note: "The original statement of the intelligence explosion: a machine that can design better machines could set off a runaway. Card 4 turns that sentence into a loop with two dials.",
    },
    {
      authors: "Vernor Vinge",
      title: "The Coming Technological Singularity: How to Survive in the Post-Human Era",
      source: "VISION-21 Symposium, NASA Lewis Research Center and Ohio Aerospace Institute",
      year: 1993,
      url: "https://edoras.sdsu.edu/~vinge/misc/singularity.html",
      note: "Names the singularity and predicts the technical means for superhuman intelligence within thirty years. It is a forecast, and a useful one to compare with the survey on card 5.",
    },
    {
      authors: "Philippe Aghion, Benjamin F. Jones, and Charles I. Jones",
      title: "Artificial Intelligence and Economic Growth",
      source: "NBER Working Paper 23928",
      year: 2017,
      url: "https://www.nber.org/papers/w23928",
      note: "Economists ask whether AI could produce a growth singularity, and argue that growth may be constrained by what is essential and hard to improve. That bottleneck is the Only people can do dial on card 4.",
    },
    {
      authors: "Ege Erdil and Tamay Besiroglu",
      title: "Explosive growth from AI automation: A review of the arguments",
      source: "arXiv:2309.11690",
      year: 2023,
      url: "https://arxiv.org/abs/2309.11690",
      note: "Weighs the arguments for and against very fast AI-driven growth and concludes it is plausible if AI can broadly replace human labour, without high confidence. It sets out the returns and bottleneck questions that card 4 turns into dials.",
    },
    {
      authors: "Katja Grace, Harlan Stewart, Julia Fabienne Sandkühler, et al.",
      title: "Thousands of AI Authors on the Future of AI",
      source: "arXiv:2401.02843",
      year: 2024,
      url: "https://arxiv.org/abs/2401.02843",
      note: "The survey on card 5: 2,778 researchers gave a 50% chance of machines beating people at every task by 2047, 13 years earlier than the previous survey, and of every occupation being automatable by 2116.",
    },
    {
      authors: "Thomas Kwa, Ben West, Joel Becker, et al.",
      title: "Measuring AI Ability to Complete Long Software Tasks",
      source: "arXiv:2503.14499",
      year: 2025,
      url: "https://arxiv.org/abs/2503.14499",
      note: "The trend on card 5: a 50% time horizon of about 50 minutes for early-2025 frontier models, doubling about every seven months since 2019, with a projection that holds only if the results generalize.",
    },
    {
      authors: "Fabrizio Dell'Acqua, Edward McFowland III, Ethan R. Mollick, et al.",
      title: "Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Effects of AI on Knowledge Worker Productivity and Quality",
      source: "Harvard Business School Working Paper 24-013; Organization Science (2026)",
      year: 2023,
      url: "https://www.semanticscholar.org/paper/409dc9f1801947f85c53231cba346f0e314cb208",
      note: "A field experiment with consultants that named the jagged frontier: AI helped on some tasks of similar difficulty and hurt on others. It is the everyday version of the jagged profile on card 2.",
    },
    {
      authors: "OpenAI",
      title: "OpenAI Charter",
      source: "openai.com",
      year: 2018,
      url: "https://openai.com/charter/",
      note: "Defines AGI as highly autonomous systems that outperform humans at most economically valuable work, one of the definitions sorted on the challenge card. It is a different bar from the two papers on cards 1 and 2.",
    },
  ],
  checkpoint: [
    {
      prompt: "AlphaFold outperforms every person at predicting protein shapes. Where does the Levels of AGI table put it?",
      options: [
        "Superhuman and narrow, so it is not ASI",
        "Superhuman and general, so it is ASI",
        "Competent and general, so it is AGI",
      ],
      answer: 0,
      explanation:
        "The table grades performance and generality separately. AlphaFold beats everyone at one task, so it sits at the top of the narrow column. ASI is the top of the general column, superhuman across a wide range of tasks, and the table marks it not yet achieved.",
      objective: 0,
    },
    {
      prompt: "On One profile, three verdicts, you switch the Rule from Add up the domains to Weakest domain decides with GPT-5 selected. What happens to the reading?",
      options: ["It stays at 57%, because the scores have not changed", "It drops to 0%, because one area scores zero", "It rises, because the strongest area now counts"],
      answer: 1,
      explanation:
        "GPT-5 scores 0 of 10 for long-term memory storage. Adding the areas gives 57%; letting the weakest area decide gives 0%. The model did not change, only the definition, which is why claims that a system is or is not AGI need to say which definition they use.",
      objective: 1,
    },
    {
      prompt: "On Faster, more, or better, 10% of a project waits on the outside world. You raise Copies from 1,000× to 1 million×. What happens to the calendar time?",
      options: ["It falls a thousandfold", "It barely changes once the world share dominates", "It rises, because copies must coordinate"],
      answer: 1,
      explanation:
        "Copies only shrink the thinking that can be split. The share that waits on the world, experiments and building, takes as long as it always did, so once thinking is fast the time sits near that floor. This toy has no coordination cost; it ignores that, rather than adding it.",
      objective: 2,
    },
    {
      prompt: "In the feedback loop on card 4, there is no human-only share and returns to research are above 1. What does the curve do after AI matches people?",
      options: ["It grows steadily, doubling at a fixed pace", "It speeds up, each doubling arriving sooner than the last", "It slows down and levels off"],
      answer: 1,
      explanation:
        "Returns above 1 mean each doubling of research effort buys more than a doubling of ability. Since the AI is doing the research, a better AI raises the effort, so the next doubling comes sooner. Returns of exactly 1 give steady doubling and returns below 1 give slowing growth.",
      objective: 3,
    },
    {
      prompt: "In the loop on card 4, you set Only people can do to 20% with returns at 1.5. Which statement is right about this lab?",
      options: [
        "It shows ASI is impossible while people do any of the work",
        "It shows a bottleneck can cap the loop, under the toy's assumptions",
        "It measures how much AI research only people can do",
      ],
      answer: 1,
      explanation:
        "With a fifth of the work at human speed, research can never run more than 5× faster than today, so in this model the loop settles into slow growth. That follows from the assumptions; it does not measure the real share, and it does not rule out ASI, since a smaller share or higher returns still reach 100×.",
      objective: 3,
    },
    {
      prompt: "A headline says experts expect human-level AI by 2047. Which reading is most accurate?",
      options: [
        "A measurement of when AI will match people",
        "A forecast whose year moves with how the question is worded",
        "A definition of what human-level AI means",
      ],
      answer: 1,
      explanation:
        "2047 is the year surveyed researchers gave a 50% chance to machines beating people at every task. The same people put every occupation being automatable at 2116, and the previous survey said 2060. The survey is measured; what it reports is a forecast, and the wording moves it.",
      objective: 4,
    },
    {
      prompt: "METR measured task length doubling about every seven months from 2019 to early 2025. What does carrying that line forward ten years tell you?",
      options: [
        "What will happen, because the trend was measured",
        "What would happen if the trend continued, which is an assumption",
        "Nothing, because past trends never continue",
      ],
      answer: 1,
      explanation:
        "The trend up to early 2025 is measured. Everything after it depends on the doubling time holding, and card 5 shows that a change from 7 to 12 months turns about 1.9 working months after five years into about 3.3 working days. A trend is evidence, not a guarantee, and it can also continue.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState,
};

export default definition;
