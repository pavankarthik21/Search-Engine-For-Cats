import { Component, OnInit } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { SearchService } from '../search.service';

@Component({
  selector: 'app-search-engine',
  templateUrl: './search-engine.component.html',
  styleUrls: ['./search-engine.component.css']
})
export class SearchEngineComponent implements OnInit {
  query: string = '';
  searchResults: any[] = [];
  searchResultsEngine: any[]=[];
  searchType:string='';
  loading: boolean = false;

  googleUrl: SafeResourceUrl | undefined;
  bingUrl: SafeResourceUrl| undefined;
  constructor(private sanitizer: DomSanitizer,
    private searchService: SearchService
  ) { }

  ngOnInit(): void {
    this.updateIframeUrls('');
  }

  search() {
    this.loading = true;
    this.searchService.searchEngine(this.query,this.searchType).subscribe(
      (response) => {
        this.searchResultsEngine = response;
        this.loading = false;
      }
    );

    this.updateIframeUrls(this.query);
    this.searchService.searchGoogle(this.query).subscribe(
      (response) => {
        this.searchResults = response.items;
      }
    );
  }

  updateIframeUrls(query: string) {
    this.googleUrl = this.sanitizer.bypassSecurityTrustResourceUrl(`https://www.google.com/search?q=${query}`);
    this.bingUrl = this.sanitizer.bypassSecurityTrustResourceUrl(`https://www.bing.com/search?q=${query}`);
  }

}
