import { useState, useEffect, useCallback } from "react";
import "@/App.css";
import axios from "axios";
import { 
  Search, 
  FileText, 
  Clock, 
  Database, 
  RefreshCw, 
  ExternalLink, 
  ChevronRight,
  Sparkles,
  BookOpen,
  TrendingUp,
  Loader2,
  AlertCircle,
  CheckCircle2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Stats Card Component
const StatCard = ({ icon: Icon, title, value, subtitle, color = "indigo" }) => {
  const colorClasses = {
    indigo: "text-indigo-400 bg-indigo-500/10",
    emerald: "text-emerald-400 bg-emerald-500/10",
    amber: "text-amber-400 bg-amber-500/10",
    blue: "text-blue-400 bg-blue-500/10"
  };

  return (
    <Card data-testid={`stat-card-${title.toLowerCase().replace(/\s+/g, '-')}`} className="stat-card glass-card-hover">
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <p className="text-sm text-zinc-500 font-medium">{title}</p>
            <p className="text-3xl font-bold font-mono tracking-tight text-zinc-100">{value}</p>
            {subtitle && (
              <p className="text-xs text-zinc-500">{subtitle}</p>
            )}
          </div>
          <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
            <Icon className="w-5 h-5" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Source Card Component
const SourceCard = ({ source, index }) => {
  return (
    <div 
      data-testid={`source-card-${index}`}
      className="group p-4 rounded-lg bg-zinc-900/50 border border-zinc-800/50 hover:border-indigo-500/30 transition-all duration-300"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <span className="badge-score">{source.relevance_score}%</span>
            <span className="font-mono text-xs text-zinc-500">{source.arxiv_id}</span>
          </div>
          <h4 className="text-sm font-medium text-zinc-200 line-clamp-2 group-hover:text-indigo-400 transition-colors">
            {source.title}
          </h4>
          <p className="text-xs text-zinc-500 mt-1 line-clamp-1">
            {source.authors?.slice(0, 3).join(", ")}{source.authors?.length > 3 ? " et al." : ""}
          </p>
        </div>
        <a 
          href={source.arxiv_url}
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 rounded-md hover:bg-zinc-800 text-zinc-500 hover:text-indigo-400 transition-all"
          data-testid={`source-link-${index}`}
        >
          <ExternalLink className="w-4 h-4" />
        </a>
      </div>
    </div>
  );
};

// Paper Card Component
const PaperCard = ({ paper, index }) => {
  const formatDate = (dateStr) => {
    try {
      return new Date(dateStr).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      });
    } catch {
      return 'Unknown date';
    }
  };

  return (
    <div 
      data-testid={`paper-card-${index}`}
      className="paper-card p-4 rounded-lg bg-zinc-900/30 border border-zinc-800/50 hover:border-indigo-500/20 transition-all duration-300"
    >
      <div className="flex items-start gap-3">
        <div className="p-2 rounded-md bg-indigo-500/10 text-indigo-400 mt-0.5">
          <FileText className="w-4 h-4" />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-medium text-zinc-200 line-clamp-2 hover:text-indigo-400 transition-colors">
            {paper.title}
          </h4>
          <p className="text-xs text-zinc-500 mt-1 line-clamp-1">
            {paper.authors?.slice(0, 3).join(", ")}{paper.authors?.length > 3 ? " et al." : ""}
          </p>
          <div className="flex items-center gap-2 mt-2 flex-wrap">
            <span className="font-mono text-[10px] text-zinc-600">{formatDate(paper.published_date)}</span>
            {paper.categories?.slice(0, 2).map((cat, i) => (
              <span key={i} className="badge-category text-[10px]">{cat}</span>
            ))}
          </div>
        </div>
        <a 
          href={paper.arxiv_url}
          target="_blank"
          rel="noopener noreferrer"
          className="p-1.5 rounded-md hover:bg-zinc-800 text-zinc-500 hover:text-indigo-400 transition-all"
        >
          <ChevronRight className="w-4 h-4" />
        </a>
      </div>
    </div>
  );
};

// Example Questions
const exampleQuestions = [
  "What are the latest advances in transformer efficiency?",
  "Explain recent breakthroughs in vision-language models",
  "What's new in reinforcement learning from human feedback?",
  "Latest techniques for reducing LLM hallucinations"
];

function App() {
  const [query, setQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState(null);
  const [stats, setStats] = useState({
    total_papers: 0,
    today_added: 0,
    last_sync: null,
    categories: [],
    sync_status: "idle"
  });
  const [recentPapers, setRecentPapers] = useState([]);
  const [isSyncing, setIsSyncing] = useState(false);
  const [activeTab, setActiveTab] = useState("search");

  // Fetch stats
  const fetchStats = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error("Failed to fetch stats:", error);
    }
  }, []);

  // Fetch recent papers
  const fetchRecentPapers = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/papers/recent?limit=20`);
      setRecentPapers(response.data);
    } catch (error) {
      console.error("Failed to fetch recent papers:", error);
    }
  }, []);

  // Initial data fetch
  useEffect(() => {
    fetchStats();
    fetchRecentPapers();
    
    // Poll for stats every 30 seconds
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, [fetchStats, fetchRecentPapers]);

  // Search handler
  const handleSearch = async (searchQuery = query) => {
    if (!searchQuery.trim()) {
      toast.error("Please enter a question");
      return;
    }

    setIsSearching(true);
    setSearchResult(null);

    try {
      const response = await axios.post(`${API}/search`, { query: searchQuery });
      setSearchResult(response.data);
      setActiveTab("search");
    } catch (error) {
      console.error("Search failed:", error);
      toast.error("Search failed. Please try again.");
    } finally {
      setIsSearching(false);
    }
  };

  // Sync handler
  const handleSync = async () => {
    if (isSyncing || stats.sync_status === "syncing") {
      toast.info("Sync is already in progress");
      return;
    }

    setIsSyncing(true);
    
    try {
      const response = await axios.post(`${API}/sync/trigger`);
      if (response.data.status === "started") {
        toast.success("Sync started! Fetching papers from arXiv...");
        // Poll for completion
        const pollInterval = setInterval(async () => {
          await fetchStats();
          const statusRes = await axios.get(`${API}/sync/status`);
          if (statusRes.data.status === "idle") {
            clearInterval(pollInterval);
            setIsSyncing(false);
            fetchRecentPapers();
            toast.success("Sync completed!");
          }
        }, 3000);
        
        // Timeout after 2 minutes
        setTimeout(() => {
          clearInterval(pollInterval);
          setIsSyncing(false);
        }, 120000);
      } else if (response.data.status === "already_running") {
        toast.info("A sync is already in progress");
        setIsSyncing(false);
      }
    } catch (error) {
      console.error("Sync failed:", error);
      toast.error("Failed to start sync");
      setIsSyncing(false);
    }
  };

  // Format last sync time
  const formatLastSync = (isoString) => {
    if (!isoString) return "Never";
    const date = new Date(isoString);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000 / 60);
    
    if (diff < 1) return "Just now";
    if (diff < 60) return `${diff}m ago`;
    if (diff < 1440) return `${Math.floor(diff / 60)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="min-h-screen" data-testid="arxiv-qa-app">
      <Toaster position="top-right" richColors />
      
      {/* Header */}
      <header className="border-b border-zinc-800/50 bg-zinc-950/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-indigo-500/10">
                <BookOpen className="w-6 h-6 text-indigo-400" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight">
                  <span className="text-gradient">arXiv</span>
                  <span className="text-zinc-100"> Research Q&A</span>
                </h1>
                <p className="text-xs text-zinc-500">AI-powered research assistant</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-zinc-900/50 border border-zinc-800/50">
                {stats.sync_status === "syncing" || isSyncing ? (
                  <>
                    <Loader2 className="w-3 h-3 text-amber-400 animate-spin" />
                    <span className="text-xs text-amber-400 font-medium">Syncing...</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                    <span className="text-xs text-zinc-400">Last sync: {formatLastSync(stats.last_sync)}</span>
                  </>
                )}
              </div>
              
              <Button
                onClick={handleSync}
                disabled={isSyncing || stats.sync_status === "syncing"}
                className="btn-secondary gap-2"
                data-testid="sync-button"
              >
                <RefreshCw className={`w-4 h-4 ${(isSyncing || stats.sync_status === "syncing") ? "animate-spin" : ""}`} />
                <span className="hidden sm:inline">Sync Papers</span>
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard
            icon={Database}
            title="Total Papers"
            value={stats.total_papers.toLocaleString()}
            subtitle="In knowledge base"
            color="indigo"
          />
          <StatCard
            icon={TrendingUp}
            title="Today Added"
            value={stats.today_added}
            subtitle="Papers ingested today"
            color="emerald"
          />
          <StatCard
            icon={Clock}
            title="Last Sync"
            value={formatLastSync(stats.last_sync)}
            subtitle="arXiv update"
            color="amber"
          />
          <StatCard
            icon={FileText}
            title="Categories"
            value={stats.categories?.length || 5}
            subtitle="AI research areas"
            color="blue"
          />
        </div>

        {/* Main Content Area */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Search & Results - Takes 8 columns */}
          <div className="lg:col-span-8 space-y-6">
            {/* Search Box */}
            <Card className="glass-card tracing-beam" data-testid="search-container">
              <CardContent className="p-6">
                <div className="search-container">
                  <div className="flex gap-3">
                    <div className="relative flex-1">
                      <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-zinc-500" />
                      <Input
                        type="text"
                        placeholder="Ask anything about AI research..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                        className="input-search pl-12 pr-4 w-full rounded-xl"
                        data-testid="search-input"
                      />
                    </div>
                    <Button
                      onClick={() => handleSearch()}
                      disabled={isSearching}
                      className="btn-primary px-6 rounded-xl gap-2"
                      data-testid="search-button"
                    >
                      {isSearching ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : (
                        <Sparkles className="w-5 h-5" />
                      )}
                      <span className="hidden sm:inline">Ask AI</span>
                    </Button>
                  </div>
                </div>

                {/* Example Questions */}
                <div className="mt-4">
                  <p className="text-xs text-zinc-500 mb-2">Try asking:</p>
                  <div className="flex flex-wrap gap-2">
                    {exampleQuestions.map((q, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          setQuery(q);
                          handleSearch(q);
                        }}
                        className="text-xs px-3 py-1.5 rounded-full bg-zinc-800/50 text-zinc-400 hover:text-indigo-400 hover:bg-zinc-800 transition-all duration-200"
                        data-testid={`example-question-${i}`}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Results Area */}
            {isSearching && (
              <Card className="glass-card">
                <CardContent className="p-8 text-center">
                  <Loader2 className="w-8 h-8 text-indigo-400 animate-spin mx-auto mb-4" />
                  <p className="text-zinc-400">Searching through research papers...</p>
                  <div className="loading-dots text-indigo-400 mt-2">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </CardContent>
              </Card>
            )}

            {searchResult && !isSearching && (
              <div className="space-y-4 animate-fade-in" data-testid="search-results">
                {/* Answer Card */}
                <Card className="glass-card">
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-indigo-400" />
                      <CardTitle className="text-lg">AI Answer</CardTitle>
                    </div>
                    <p className="text-xs text-zinc-500 font-mono mt-1">
                      Query: "{searchResult.query}"
                    </p>
                  </CardHeader>
                  <CardContent>
                    <div className="answer-text text-zinc-300 leading-relaxed whitespace-pre-wrap" data-testid="answer-text">
                      {searchResult.answer}
                    </div>
                  </CardContent>
                </Card>

                {/* Sources Card */}
                {searchResult.sources && searchResult.sources.length > 0 && (
                  <Card className="glass-card">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <FileText className="w-5 h-5 text-emerald-400" />
                          <CardTitle className="text-lg">Sources</CardTitle>
                        </div>
                        <Badge variant="outline" className="text-xs">
                          {searchResult.sources.length} papers
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3" data-testid="sources-list">
                        {searchResult.sources.map((source, index) => (
                          <SourceCard key={source.arxiv_id} source={source} index={index} />
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            )}

            {!searchResult && !isSearching && (
              <Card className="glass-card border-dashed">
                <CardContent className="p-12 text-center">
                  <div className="p-4 rounded-full bg-zinc-800/50 w-fit mx-auto mb-4">
                    <Search className="w-8 h-8 text-zinc-600" />
                  </div>
                  <h3 className="text-lg font-medium text-zinc-400 mb-2">
                    Ask a question about AI research
                  </h3>
                  <p className="text-sm text-zinc-500 max-w-md mx-auto">
                    Our AI will search through the latest arXiv papers and provide you with a comprehensive answer backed by sources.
                  </p>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Recent Papers Sidebar - Takes 4 columns */}
          <div className="lg:col-span-4">
            <Card className="glass-card h-fit" data-testid="recent-papers-card">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Clock className="w-5 h-5 text-amber-400" />
                    <CardTitle className="text-lg">Recent Papers</CardTitle>
                  </div>
                  <Badge variant="outline" className="text-xs font-mono">
                    {recentPapers.length}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <ScrollArea className="h-[600px]">
                  <div className="space-y-2 p-4 pt-0">
                    {recentPapers.length > 0 ? (
                      recentPapers.map((paper, index) => (
                        <PaperCard key={paper.arxiv_id || index} paper={paper} index={index} />
                      ))
                    ) : (
                      <div className="text-center py-8">
                        <AlertCircle className="w-8 h-8 text-zinc-600 mx-auto mb-2" />
                        <p className="text-sm text-zinc-500">No papers yet</p>
                        <p className="text-xs text-zinc-600 mt-1">Click "Sync Papers" to fetch from arXiv</p>
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Categories Footer */}
        <div className="mt-8 pt-8 border-t border-zinc-800/50">
          <div className="flex flex-wrap items-center justify-center gap-3">
            <span className="text-xs text-zinc-500">Monitoring:</span>
            {stats.categories?.map((cat, i) => (
              <span key={i} className="badge-category">{cat}</span>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800/50 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-xs text-zinc-500">
              Powered by arXiv, ChromaDB, and Claude AI
            </p>
            <div className="flex items-center gap-4">
              <a 
                href="https://arxiv.org" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-xs text-zinc-500 hover:text-indigo-400 transition-colors"
              >
                arXiv.org
              </a>
              <span className="text-zinc-700">•</span>
              <span className="text-xs text-zinc-600 font-mono">v1.0.0</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
