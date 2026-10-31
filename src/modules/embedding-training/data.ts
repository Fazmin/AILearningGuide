/**
 * The authored world of this lab: 41 short notices from the Gull Harbor office (the same harbor as
 * Embeddings for search and Retrieval-augmented generation), the training queries a support desk
 * might have logged against them, and a separate held-out set that training never sees.
 *
 * Everything here is written by hand. The passages fall into ten topics of three to five notices that
 * are easy to confuse (three docks with different hours, five prices, four sailings). Three facts are
 * stated twice in different words, as real collections often do; each query labels only one member of
 * such a pair as its answer, the way sparse relevance labels usually work, and evaluation counts both.
 *
 * Queries deliberately paraphrase: they say jetty for dock, shut for close, boat or crossing for ferry,
 * price for costs. Keyword search therefore has to make do with the words a query happens to share
 * with a notice, while a trained encoder can learn the paraphrases from the training pairs.
 */

export interface Topic {
  id: string;
  label: string;
}

export const TOPICS: readonly Topic[] = [
  { id: "docks", label: "Dock hours" },
  { id: "sailings", label: "Sailings" },
  { id: "prices", label: "Prices" },
  { id: "animals", label: "Animals" },
  { id: "weather", label: "Weather" },
  { id: "safety", label: "Safety" },
  { id: "lost", label: "Lost and stored" },
  { id: "fishing", label: "Fishing" },
  { id: "access", label: "Getting here" },
  { id: "office", label: "Office" },
];

export interface Passage {
  id: string;
  title: string;
  text: string;
  topic: string;
}

export const PASSAGES: readonly Passage[] = [
  { id: "north-hours", title: "North dock hours", topic: "docks", text: "The north dock opens at 06:00 and closes at 18:00 on weekdays." },
  { id: "south-hours", title: "South dock hours", topic: "docks", text: "The south dock stays open until 21:00 every day of the week." },
  { id: "east-hours", title: "East dock hours", topic: "docks", text: "The east dock is open from 08:00 to 17:00, Monday to Saturday." },
  { id: "sunday-hours", title: "Sunday hours", topic: "docks", text: "On Sundays the north dock closes early, at 16:00." },
  { id: "sunday-notice", title: "Sunday notice", topic: "docks", text: "Notice: every Sunday the north dock shuts at four in the afternoon." },

  { id: "ferry-morning", title: "Morning ferry", topic: "sailings", text: "The island ferry leaves the harbor at 07:15 every morning." },
  { id: "ferry-afternoon", title: "Afternoon ferry", topic: "sailings", text: "The afternoon ferry to the island departs at 15:40." },
  { id: "ferry-return", title: "Last ferry back", topic: "sailings", text: "The last ferry back from the island sails at 19:30." },
  { id: "ferry-winter", title: "Winter timetable", topic: "sailings", text: "From November to March the ferry runs only at weekends." },
  { id: "ferry-crossing", title: "Crossing time", topic: "sailings", text: "The crossing to the island takes forty minutes in calm weather." },

  { id: "ferry-fares", title: "Ferry fares", topic: "prices", text: "A single ferry ticket costs 12 tokens and children ride free." },
  { id: "ticket-desk", title: "Ticket desk", topic: "prices", text: "One-way crossings are 12 tokens for adults; under-twelves travel free." },
  { id: "mooring-fees", title: "Mooring fees", topic: "prices", text: "Mooring a boat overnight at the south dock costs 5 tokens." },
  { id: "parking-fees", title: "Parking fees", topic: "prices", text: "Parking in the harbor car park costs 3 tokens per day." },
  { id: "slipway-fee", title: "Slipway fee", topic: "prices", text: "Launching a small boat from the slipway costs 2 tokens." },

  { id: "pets-board", title: "Pets on board", topic: "animals", text: "Pets may travel on the ferry on a leash and must wear a tag." },
  { id: "dogs-quay", title: "Dogs on the quay", topic: "animals", text: "Dogs are welcome on the quay but not on the fishing pier." },
  { id: "gulls", title: "Feeding wildlife", topic: "animals", text: "Please do not feed the gulls or the seals in the harbor." },

  { id: "storm-order", title: "Storm order", topic: "weather", text: "If wind exceeds 40 knots the harbor master closes both docks." },
  { id: "fog-rule", title: "Fog rule", topic: "weather", text: "In fog the ferry sounds its horn and crosses slowly." },
  { id: "frost-warning", title: "Frost warning", topic: "weather", text: "In frost the ramps and steps get slippery, so hold the handrails." },
  { id: "tide-board", title: "Tide board", topic: "weather", text: "High tide times are posted each morning on the board outside the office." },

  { id: "life-jackets", title: "Life jackets", topic: "safety", text: "Life jackets for children and adults are stored in shed B." },
  { id: "flares", title: "Flares", topic: "safety", text: "Distress flares are kept in the red box beside the fuel pump." },
  { id: "first-aid", title: "First aid", topic: "safety", text: "A first aid kit and a defibrillator hang inside the office door." },
  { id: "fuel-safety", title: "Fuel safety", topic: "safety", text: "No smoking or open flames within ten metres of the fuel pump." },

  { id: "lost-property", title: "Lost property", topic: "lost", text: "Lost keys, bags and phones are kept at the harbor office." },
  { id: "found-items", title: "Found items", topic: "lost", text: "Anything found on the quay is held at the harbor office until collected." },
  { id: "left-luggage", title: "Left luggage", topic: "lost", text: "Bags can be left in the luggage room for 1 token per day." },
  { id: "lost-ticket", title: "Lost tickets", topic: "lost", text: "A lost ferry ticket cannot be refunded or replaced." },

  { id: "fishing-pier", title: "Fishing", topic: "fishing", text: "Fishing from the north pier is allowed before dawn." },
  { id: "fishing-licence", title: "Fishing licence", topic: "fishing", text: "Anglers need a licence, sold at the harbor office for 4 tokens." },
  { id: "crab-pots", title: "Crab pots", topic: "fishing", text: "Crab pots must carry a tag with the owner's mooring number." },

  { id: "car-park", title: "Car park hours", topic: "access", text: "The harbor car park gates are locked at midnight." },
  { id: "bike-racks", title: "Bicycle racks", topic: "access", text: "Bicycle racks stand next to the ticket office." },
  { id: "step-free", title: "Step-free access", topic: "access", text: "Wheelchair users can board the ferry by the ramp at the east dock." },
  { id: "bus-stop", title: "Bus stop", topic: "access", text: "The town bus stops outside the harbor gates every half hour." },

  { id: "office-hours", title: "Office hours", topic: "office", text: "The harbor office is open from 09:00 to 17:00, Monday to Friday." },
  { id: "showers", title: "Showers", topic: "office", text: "Hot showers for boat crews are in the block behind the office." },
  { id: "wifi", title: "Wi-Fi", topic: "office", text: "Free wireless internet is available in the waiting room." },
  { id: "waiting-room", title: "Waiting room", topic: "office", text: "The waiting room opens one hour before each ferry departure." },
];

/** The same fact, written twice. Evaluation counts both members as relevant to a query that labels either. */
export const DUPLICATE_PAIRS: readonly (readonly [string, string])[] = [
  ["sunday-hours", "sunday-notice"],
  ["ferry-fares", "ticket-desk"],
  ["lost-property", "found-items"],
];

export interface Query {
  id: string;
  text: string;
  /** The one passage this query's label names. */
  answer: string;
}

export type HeldOutKind = "paraphrase" | "typo" | "unseen";

export interface HeldOutQuery extends Query {
  kind: HeldOutKind;
}

const train = (id: string, text: string, answer: string): Query => ({ id, text, answer });

/** 90 training queries: what the encoder learns from. Each names one labelled answer. */
export const TRAIN_QUERIES: readonly Query[] = [
  train("t01", "When does the north jetty shut on weekdays?", "north-hours"),
  train("t02", "north berth opening time on a weekday", "north-hours"),
  train("t03", "Can I get onto the north jetty at 7 in the morning?", "north-hours"),
  train("t04", "How late is the south jetty open?", "south-hours"),
  train("t05", "south berth closing time", "south-hours"),
  train("t06", "Can I use the south jetty at night?", "south-hours"),
  train("t07", "When does the east jetty open?", "east-hours"),
  train("t08", "Is the east berth open on Saturday?", "east-hours"),
  train("t09", "east jetty hours", "east-hours"),
  train("t10", "Does the north jetty close early at the weekend?", "sunday-hours"),
  train("t11", "north berth Sunday closing time", "sunday-hours"),
  train("t12", "Sunday early closing at the north jetty", "sunday-notice"),
  train("t13", "Is the north berth open late on a Sunday?", "sunday-notice"),

  train("t14", "first boat to the island", "ferry-morning"),
  train("t15", "What time does the early crossing go?", "ferry-morning"),
  train("t16", "morning sailing to the island", "ferry-morning"),
  train("t17", "When does the afternoon boat go?", "ferry-afternoon"),
  train("t18", "afternoon sailing time", "ferry-afternoon"),
  train("t19", "last boat home from the island", "ferry-return"),
  train("t20", "When is the final sailing back?", "ferry-return"),
  train("t21", "Does the boat run in January?", "ferry-winter"),
  train("t22", "winter sailings to the island", "ferry-winter"),
  train("t23", "How long is the boat trip?", "ferry-crossing"),
  train("t24", "how many minutes does the crossing take", "ferry-crossing"),

  train("t25", "How much is a boat ticket?", "ferry-fares"),
  train("t26", "price of a crossing for kids", "ferry-fares"),
  train("t27", "adult fare to the island", "ticket-desk"),
  train("t28", "Do children pay for the boat?", "ticket-desk"),
  train("t29", "How much to keep my yacht at the south jetty overnight?", "mooring-fees"),
  train("t30", "overnight berth price", "mooring-fees"),
  train("t31", "How much does it cost to leave my car all day?", "parking-fees"),
  train("t32", "daily car charge", "parking-fees"),
  train("t33", "price to put a dinghy in the water", "slipway-fee"),
  train("t34", "launch ramp charge for a small boat", "slipway-fee"),

  train("t35", "Can my dog come on the boat?", "pets-board"),
  train("t36", "Are cats allowed on the crossing?", "pets-board"),
  train("t37", "Can I walk my dog along the waterfront?", "dogs-quay"),
  train("t38", "dogs near the fishing area", "dogs-quay"),
  train("t39", "Can I give bread to the seagulls?", "gulls"),
  train("t40", "feeding seals", "gulls"),

  train("t41", "Is the harbour shut in a gale?", "storm-order"),
  train("t42", "What happens when it is very windy?", "storm-order"),
  train("t43", "What does the boat do in mist?", "fog-rule"),
  train("t44", "foggy crossing rules", "fog-rule"),
  train("t45", "Are the steps icy in winter?", "frost-warning"),
  train("t46", "slippery ramps when it freezes", "frost-warning"),
  train("t47", "Where can I find high water times?", "tide-board"),
  train("t48", "tide times today", "tide-board"),

  train("t49", "Where are the life vests kept?", "life-jackets"),
  train("t50", "buoyancy aids for kids", "life-jackets"),
  train("t51", "Where is the emergency rocket box?", "flares"),
  train("t52", "distress signal kit location", "flares"),
  train("t53", "Where can I get a plaster for a cut?", "first-aid"),
  train("t54", "Is there a defib at the harbour?", "first-aid"),
  train("t55", "Can I smoke near the petrol pump?", "fuel-safety"),
  train("t56", "fire rules at the diesel pump", "fuel-safety"),

  train("t57", "I left my wallet on the boat, who has it?", "lost-property"),
  train("t58", "lost phone", "lost-property"),
  train("t59", "Where do I hand in a purse I picked up?", "found-items"),
  train("t60", "someone found my keys on the waterfront", "found-items"),
  train("t61", "Can I store my suitcase for the day?", "left-luggage"),
  train("t62", "bag storage price", "left-luggage"),
  train("t63", "I lost my boat ticket, can I get a new one?", "lost-ticket"),
  train("t64", "refund for a missing ticket", "lost-ticket"),

  train("t65", "Can I go angling from the north jetty early?", "fishing-pier"),
  train("t66", "when is fishing allowed", "fishing-pier"),
  train("t67", "Do I need a permit to fish?", "fishing-licence"),
  train("t68", "angling permit price", "fishing-licence"),
  train("t69", "rules for lobster creels", "crab-pots"),
  train("t70", "how to label crab traps", "crab-pots"),

  train("t71", "What time does the car park close at night?", "car-park"),
  train("t72", "Can I collect my car late in the evening?", "car-park"),
  train("t73", "Where can I leave my bike?", "bike-racks"),
  train("t74", "cycle parking", "bike-racks"),
  train("t75", "wheelchair access to the boat", "step-free"),
  train("t76", "Is there a ramp for prams?", "step-free"),
  train("t77", "How often does the bus come?", "bus-stop"),
  train("t78", "bus to town from the harbour", "bus-stop"),

  train("t79", "When is the harbour office open?", "office-hours"),
  train("t80", "office opening times on Friday", "office-hours"),
  train("t81", "Where can sailors wash?", "showers"),
  train("t82", "hot water for crews", "showers"),
  train("t83", "Is there free internet?", "wifi"),
  train("t84", "Can I get online while I wait?", "wifi"),
  train("t85", "When does the lounge open before the boat?", "waiting-room"),
  train("t86", "Where can I sit while waiting for the crossing?", "waiting-room"),

  train("t87", "Is the south jetty shut on Sundays?", "south-hours"),
  train("t88", "east berth closing time", "east-hours"),
  train("t89", "Is the harbour closed when the wind is strong?", "storm-order"),
  train("t90", "keys left on the boat", "lost-property"),
];

const held = (id: string, text: string, answer: string, kind: HeldOutKind): HeldOutQuery => ({ id, text, answer, kind });

/**
 * 30 held-out queries, never trained on. Twenty paraphrase with words the training queries use, in
 * new combinations; five carry typos or run words together; five lean on words no training query or
 * notice contains, which no amount of training on this set can teach.
 */
export const HELD_OUT_QUERIES: readonly HeldOutQuery[] = [
  held("h01", "When does the north jetty shut on Sundays?", "sunday-hours", "paraphrase"),
  held("h02", "How late can I use the east berth?", "east-hours", "paraphrase"),
  held("h03", "south berth opening hours on a Sunday", "south-hours", "paraphrase"),
  held("h04", "What time does the first sailing leave?", "ferry-morning", "paraphrase"),
  held("h05", "When does the last crossing come back?", "ferry-return", "paraphrase"),
  held("h06", "afternoon boat to the island", "ferry-afternoon", "paraphrase"),
  held("h07", "How long does the sailing take?", "ferry-crossing", "paraphrase"),
  held("h08", "Does the crossing run in winter?", "ferry-winter", "paraphrase"),
  held("h09", "What do kids pay for the crossing?", "ferry-fares", "paraphrase"),
  held("h10", "price to leave a yacht overnight", "mooring-fees", "paraphrase"),
  held("h11", "daily charge to park", "parking-fees", "paraphrase"),
  held("h12", "Can a cat come on the crossing?", "pets-board", "paraphrase"),
  held("h13", "Is the harbour shut when it is windy?", "storm-order", "paraphrase"),
  held("h14", "What does the boat do when it is foggy?", "fog-rule", "paraphrase"),
  held("h15", "Where are buoyancy aids stored?", "life-jackets", "paraphrase"),
  held("h16", "Can I smoke near the diesel pump?", "fuel-safety", "paraphrase"),
  held("h17", "I found a wallet on the waterfront", "found-items", "paraphrase"),
  held("h18", "Do I need an angling permit?", "fishing-licence", "paraphrase"),
  held("h19", "Where can I leave my bicycle?", "bike-racks", "paraphrase"),
  held("h20", "Can crews get a hot wash?", "showers", "paraphrase"),

  held("h21", "wen dose the norht jetty shut", "north-hours", "typo"),
  held("h22", "feries to the iland in winter", "ferry-winter", "typo"),
  held("h23", "lifevests for kids", "life-jackets", "typo"),
  held("h24", "carpark gates at nite", "car-park", "typo"),
  held("h25", "harbour ofice opening times", "office-hours", "typo"),

  held("h26", "Can my hound ride the vessel?", "pets-board", "unseen"),
  held("h27", "Is the marina closed during a tempest?", "storm-order", "unseen"),
  held("h28", "Do toddlers sail for nothing?", "ferry-fares", "unseen"),
  held("h29", "When does the catamaran depart at dawn?", "ferry-morning", "unseen"),
  held("h30", "Can I stash my rucksack somewhere?", "left-luggage", "unseen"),
];

export const DEFAULT_QUERY_ID = "h01";

export const PASSAGE_INDEX = new Map(PASSAGES.map((passage, index) => [passage.id, index]));

export const passageIndex = (id: string) => {
  const index = PASSAGE_INDEX.get(id);
  if (index === undefined) throw new Error(`Unknown passage ${id}`);
  return index;
};

/** For each passage index, the index of its duplicate, or −1. */
export const DUPLICATE_OF: readonly number[] = (() => {
  const twins = PASSAGES.map(() => -1);
  for (const [a, b] of DUPLICATE_PAIRS) {
    twins[passageIndex(a)] = passageIndex(b);
    twins[passageIndex(b)] = passageIndex(a);
  }
  return twins;
})();

/** Every passage that answers a query: its labelled answer and that answer's duplicate, if any. */
export function relevantFor(query: Query): number[] {
  const answer = passageIndex(query.answer);
  return DUPLICATE_OF[answer] >= 0 ? [answer, DUPLICATE_OF[answer]] : [answer];
}

export const HELD_OUT_KIND_LABEL: Record<HeldOutKind, string> = {
  paraphrase: "paraphrase",
  typo: "typo",
  unseen: "unseen words",
};
