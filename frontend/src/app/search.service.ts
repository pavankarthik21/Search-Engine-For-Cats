import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root'
})
export class SearchService {
  private apiServerUrl = environment.apiBaseURl;
  // private apiKey = 'AIzaSyBbg2CDUkcWkLJkAk8NM538TICrd4jm-ok';
  private apiKey = 'AIzaSyCUb9EmajbLYeFZtolnAkkRsJJwdlS9uHs';
  private searchEngineId = 'd782f2779ef8d48bc';
  private apiUrl = 'https://www.googleapis.com/customsearch/v1';

  constructor(private http: HttpClient) {}

  public searchGoogle(query: string): Observable<any> {
    const params = new HttpParams()
      .set('key', this.apiKey)
      .set('cx', this.searchEngineId)
      .set('q', query);
    return this.http.get<any>(this.apiUrl, { params }); 
  }

  public searchEngine(query:String,searchType:String):Observable<any>{
    return this.http.get<any>(`${this.apiServerUrl}/index?query=${query}&type=${searchType}`);
   }


}
